"""Run RLHF capture using an external transcript (Premiere Pro SRT, Frame.io VTT, or plain text).
Skips audio extraction and Whisper entirely.

Usage:
  python3 rlhf_from_transcript.py path/to/transcript.srt
  python3 rlhf_from_transcript.py path/to/transcript.vtt
  python3 rlhf_from_transcript.py path/to/transcript.txt
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from processor import (
    _format_for_llm,
    _parse_line_numbers,
    _get_client,
    PASS_1_PROMPT,
    PASS_2_PROMPT,
    PASS_3_PROMPT,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "rlhf")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Transcript parsers
# ---------------------------------------------------------------------------

def parse_srt(path: str) -> list[dict]:
    """Parse SRT file into numbered lines with timestamps."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []

    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) < 3:
            continue

        # Line 1: sequence number (ignore, we renumber)
        # Line 2: timecodes  00:01:23,456 --> 00:01:25,789
        # Line 3+: text
        tc_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            block_lines[1].strip(),
        )
        if not tc_match:
            continue

        g = tc_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000

        text = " ".join(block_lines[2:]).strip()
        # Strip HTML tags (Premiere sometimes adds <i>, <b>, etc.)
        text = re.sub(r"<[^>]+>", "", text).strip()

        if text:
            lines.append({
                "id": len(lines) + 1,
                "start": start,
                "end": end,
                "text": text,
            })

    return lines


def parse_vtt(path: str) -> list[dict]:
    """Parse WebVTT file into numbered lines with timestamps."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip WEBVTT header and any metadata
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    # Strip NOTE blocks
    content = re.sub(r"NOTE.*?\n\n", "", content, flags=re.DOTALL)

    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []

    for block in blocks:
        block_lines = block.strip().split("\n")

        # Find the timecode line (might have optional cue ID before it)
        tc_line = None
        text_start = 0
        for i, bl in enumerate(block_lines):
            if "-->" in bl:
                tc_line = bl
                text_start = i + 1
                break

        if not tc_line:
            continue

        # VTT timecodes: 00:01:23.456 --> 00:01:25.789  or  01:23.456 --> 01:25.789
        tc_match = re.match(
            r"(?:(\d{2}):)?(\d{2}):(\d{2})[.](\d{3})\s*-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})[.](\d{3})",
            tc_line.strip(),
        )
        if not tc_match:
            continue

        g = tc_match.groups()
        start = int(g[0] or 0) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4] or 0) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000

        text = " ".join(block_lines[text_start:]).strip()
        text = re.sub(r"<[^>]+>", "", text).strip()

        if text:
            lines.append({
                "id": len(lines) + 1,
                "start": start,
                "end": end,
                "text": text,
            })

    return lines


def parse_plain_text(path: str) -> list[dict]:
    """Parse plain text transcript (no timestamps). Assigns fake 3s-per-line timing."""
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    lines = []
    t = 0.0
    for text in raw_lines:
        # Estimate ~3 seconds per line
        duration = max(1.0, len(text.split()) * 0.3)
        lines.append({
            "id": len(lines) + 1,
            "start": t,
            "end": t + duration,
            "text": text,
        })
        t += duration + 0.2

    return lines


def parse_premiere_txt(path: str, fps: float = 29.97) -> list[dict]:
    """Parse Premiere Pro text transcript export.

    Format:
        00:00:03:05 - 00:00:10:03
        Unknown
        Michael. Yeah.

    Timecodes are HH:MM:SS:FF (frames, not milliseconds).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []

    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) < 3:
            continue

        # Line 1: timecodes  00:00:03:05 - 00:00:10:03
        tc_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2}):(\d{2}):(\d{2})",
            block_lines[0].strip(),
        )
        if not tc_match:
            continue

        g = tc_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / fps
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / fps

        # Line 2: speaker label (skip)
        # Line 3+: text
        text = " ".join(block_lines[2:]).strip()
        text = re.sub(r"<[^>]+>", "", text).strip()

        if text:
            lines.append({
                "id": len(lines) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            })

    return lines


def detect_and_parse(path: str) -> list[dict]:
    """Auto-detect format and parse."""
    ext = os.path.splitext(path)[1].lower()

    # Peek at content to auto-detect
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(500)

    # Premiere Pro TXT: "00:00:03:05 - 00:00:10:03" (frame-based, dash separator)
    if re.search(r"\d{2}:\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:\d{2}", head):
        print(f"  Detected Premiere Pro transcript format (frame-based timecodes)")
        return parse_premiere_txt(path)

    if ext == ".srt":
        print(f"  Detected SRT format")
        return parse_srt(path)
    elif ext == ".vtt":
        print(f"  Detected WebVTT format")
        return parse_vtt(path)
    elif "WEBVTT" in head:
        print(f"  Detected WebVTT format (from content)")
        return parse_vtt(path)
    elif re.search(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", head):
        print(f"  Detected SRT format (from content)")
        return parse_srt(path)
    else:
        print(f"  Detected plain text format (no timestamps — timing will be estimated)")
        return parse_plain_text(path)


# ---------------------------------------------------------------------------
# LLM passes (same as rlhf_capture.py)
# ---------------------------------------------------------------------------

def run_passes(lines, total_duration):
    print(f"Running 3 passes through GPT-4o...")
    client = _get_client()
    valid_ids = {l["id"] for l in lines}
    full_transcript = _format_for_llm(lines)
    results = {}

    # Pass 1
    print(f"\n  [Pass 1] {len(lines)} lines → GPT-4o...")
    t0 = time.time()
    r1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_1_PROMPT},
            {"role": "user", "content": full_transcript},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    keep_1 = [k for k in _parse_line_numbers(r1.choices[0].message.content) if k in valid_ids]
    dur_1 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_1))
    print(f"  [Pass 1] {len(keep_1)}/{len(lines)} lines kept ({dur_1:.0f}s, {dur_1/total_duration*100:.0f}%) — {time.time()-t0:.0f}s")
    results["pass_1"] = {"kept": keep_1, "raw_response": r1.choices[0].message.content}

    with open(os.path.join(OUTPUT_DIR, "pass1_raw.txt"), "w") as f:
        f.write(r1.choices[0].message.content)

    if not keep_1:
        raise RuntimeError("Pass 1 returned no lines")

    # Pass 2
    survivors_1 = [l for l in lines if l["id"] in set(keep_1)]
    print(f"\n  [Pass 2] {len(survivors_1)} survivors → GPT-4o...")
    t0 = time.time()
    r2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_2_PROMPT},
            {"role": "user", "content": _format_for_llm(survivors_1)},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    keep_2 = [k for k in _parse_line_numbers(r2.choices[0].message.content) if k in valid_ids]
    if not keep_2:
        print("  Warning: Pass 2 empty, using Pass 1")
        keep_2 = keep_1
    dur_2 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_2))
    print(f"  [Pass 2] {len(keep_2)}/{len(survivors_1)} lines kept ({dur_2:.0f}s, {dur_2/total_duration*100:.0f}%) — {time.time()-t0:.0f}s")
    results["pass_2"] = {"kept": keep_2, "raw_response": r2.choices[0].message.content}

    with open(os.path.join(OUTPUT_DIR, "pass2_raw.txt"), "w") as f:
        f.write(r2.choices[0].message.content)

    # Pass 3
    survivors_2 = [l for l in lines if l["id"] in set(keep_2)]
    print(f"\n  [Pass 3] {len(survivors_2)} survivors → GPT-4o...")
    t0 = time.time()
    r3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_3_PROMPT},
            {"role": "user", "content": _format_for_llm(survivors_2)},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    keep_3 = [k for k in _parse_line_numbers(r3.choices[0].message.content) if k in valid_ids]
    if not keep_3:
        print("  Warning: Pass 3 empty, using Pass 2")
        keep_3 = keep_2
    dur_3 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_3))
    print(f"  [Pass 3] {len(keep_3)}/{len(survivors_2)} lines kept ({dur_3:.0f}s, {dur_3/total_duration*100:.0f}%) — {time.time()-t0:.0f}s")
    results["pass_3"] = {"kept": keep_3, "raw_response": r3.choices[0].message.content}

    with open(os.path.join(OUTPUT_DIR, "pass3_raw.txt"), "w") as f:
        f.write(r3.choices[0].message.content)

    return results


# ---------------------------------------------------------------------------
# RLHF review file builder
# ---------------------------------------------------------------------------

def build_rlhf_file(lines, results, total_duration, source_label):
    keep_1 = set(results["pass_1"]["kept"])
    keep_2 = set(results["pass_2"]["kept"])
    keep_3 = set(results["pass_3"]["kept"])

    dur_1 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_1)
    dur_2 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_2)
    dur_3 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_3)

    out = []
    out.append("=" * 90)
    out.append(f"RLHF REVIEW — Project C5296 (transcript source: {source_label})")
    out.append("=" * 90)
    out.append("")
    out.append("HOW TO USE THIS FILE:")
    out.append("  Each line shows: line number, timestamps, the LLM's decision, and the text.")
    out.append("  Decision key:")
    out.append("    KEEP     = survived all 3 passes (in final cut)")
    out.append("    CUT@P1   = cut in Pass 1 (initial aggressive cut)")
    out.append("    CUT@P2   = survived Pass 1, cut in Pass 2 (false start / repeat hunting)")
    out.append("    CUT@P3   = survived Pass 1+2, cut in Pass 3 (final QC)")
    out.append("")
    out.append("  ADD YOUR FEEDBACK after each line using -->")
    out.append("  Examples:")
    out.append("    --> WRONG: should be CUT, this is a false start")
    out.append("    --> WRONG: should be KEEP, this is clean delivery")
    out.append("    --> WRONG: duplicate of L0234, keep that one instead (it's the later take)")
    out.append("    --> OK")
    out.append("    --> this whole section (L0100-L0120) is a redo of L0050-L0070, keep L0100-L0120")
    out.append("")
    out.append(f"STATS:")
    out.append(f"  Total lines: {len(lines)}")
    out.append(f"  Raw duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    out.append(f"  After Pass 1: {len(keep_1)} lines, {dur_1:.0f}s ({dur_1/total_duration*100:.0f}%)")
    out.append(f"  After Pass 2: {len(keep_2)} lines, {dur_2:.0f}s ({dur_2/total_duration*100:.0f}%)")
    out.append(f"  After Pass 3: {len(keep_3)} lines, {dur_3:.0f}s ({dur_3/total_duration*100:.0f}%)")
    out.append(f"  Target: ~45% (~{total_duration*0.45:.0f}s, ~{total_duration*0.45/60:.1f} min)")
    out.append("")
    out.append("=" * 90)
    out.append("TRANSCRIPT + DECISIONS")
    out.append("=" * 90)
    out.append("")

    for line in lines:
        lid = line["id"]
        ts = f"[{line['start']:.1f}s-{line['end']:.1f}s]"

        if lid in keep_3:
            decision = "KEEP  "
        elif lid in keep_2:
            decision = "CUT@P3"
        elif lid in keep_1:
            decision = "CUT@P2"
        else:
            decision = "CUT@P1"

        out.append(f"L{lid:04d} {ts:20s} {decision}  {line['text']}")
        out.append(f"  -->")
        out.append("")

    path = os.path.join(OUTPUT_DIR, "rlhf_review.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"\n  RLHF review file: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rlhf_from_transcript.py <transcript_file>")
        print("  Supports: .srt, .vtt, .txt")
        sys.exit(1)

    transcript_path = sys.argv[1]
    if not os.path.exists(transcript_path):
        print(f"File not found: {transcript_path}")
        sys.exit(1)

    print(f"[1/3] Parsing transcript: {transcript_path}")
    lines = detect_and_parse(transcript_path)
    total_duration = lines[-1]["end"] if lines else 0
    print(f"  {len(lines)} lines, {total_duration:.0f}s ({total_duration/60:.1f} min)")

    # Save parsed lines as JSON for reuse
    parsed_path = os.path.join(OUTPUT_DIR, "parsed_lines.json")
    with open(parsed_path, "w") as f:
        json.dump(lines, f, indent=2)
    print(f"  Saved parsed lines to {parsed_path}")

    # Save numbered transcript
    transcript_out = os.path.join(OUTPUT_DIR, "numbered_transcript.txt")
    with open(transcript_out, "w") as f:
        f.write(_format_for_llm(lines))
    print(f"  Saved numbered transcript to {transcript_out}")

    print(f"\n[2/3] Running LLM passes...")
    results = run_passes(lines, total_duration)

    source_label = os.path.basename(transcript_path)
    print(f"\n[3/3] Building RLHF review file...")
    build_rlhf_file(lines, results, total_duration, source_label)

    print("\nDone. Review the file, add --> comments, then I'll rebuild the prompts.")


if __name__ == "__main__":
    main()
