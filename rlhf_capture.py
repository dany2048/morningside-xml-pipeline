"""Run pipeline on test file, capture all intermediates for RLHF review."""
from __future__ import annotations

import json
import os
import sys
import time

# Add pipeline dir to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from audio import extract_audio, chunk_audio
from transcribe import transcribe_all
from processor import (
    _build_numbered_lines,
    _format_for_llm,
    _parse_line_numbers,
    _get_client,
    PASS_1_PROMPT,
    PASS_2_PROMPT,
    PASS_3_PROMPT,
)

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "reference",
    "test file for cutting project C5296 .MP4"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "rlhf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WORDS_CACHE = os.path.join(OUTPUT_DIR, "c5296_words.json")
AUDIO_CACHE = os.path.join(OUTPUT_DIR, "c5296_audio.mp3")


def step_1_extract_audio():
    if os.path.exists(AUDIO_CACHE):
        print(f"[1/4] Audio cached at {AUDIO_CACHE}")
        return AUDIO_CACHE
    print(f"[1/4] Extracting audio from 33GB file...")
    t0 = time.time()
    audio_path = extract_audio(INPUT_FILE, AUDIO_CACHE)
    print(f"  Done in {time.time()-t0:.0f}s")
    return audio_path


def step_2_transcribe(audio_path):
    if os.path.exists(WORDS_CACHE):
        print(f"[2/4] Transcript cached at {WORDS_CACHE}")
        with open(WORDS_CACHE) as f:
            return json.load(f)
    print(f"[2/4] Chunking + transcribing with Whisper API...")
    t0 = time.time()
    chunk_dir = os.path.join(OUTPUT_DIR, "chunks")
    chunks = chunk_audio(audio_path, chunk_dir)
    words = transcribe_all(chunks)
    with open(WORDS_CACHE, "w") as f:
        json.dump(words, f)
    print(f"  Done in {time.time()-t0:.0f}s — {len(words)} words")
    return words


def step_3_build_lines(words):
    print(f"[3/4] Building numbered lines...")
    lines = _build_numbered_lines(words)
    # Save raw transcript
    transcript_path = os.path.join(OUTPUT_DIR, "c5296_numbered_transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(_format_for_llm(lines))
    print(f"  {len(lines)} lines saved to {transcript_path}")
    return lines


def step_4_run_passes(lines, total_duration):
    print(f"[4/4] Running 3 passes through GPT-4o...")
    client = _get_client()
    valid_ids = {l["id"] for l in lines}
    full_transcript = _format_for_llm(lines)
    results = {}

    # Pass 1
    print("\n  [Pass 1] Full transcript → GPT-4o...")
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

    # Save Pass 1 raw response
    with open(os.path.join(OUTPUT_DIR, "c5296_pass1_raw.txt"), "w") as f:
        f.write(r1.choices[0].message.content)

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
        keep_2 = keep_1
    dur_2 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_2))
    print(f"  [Pass 2] {len(keep_2)}/{len(survivors_1)} lines kept ({dur_2:.0f}s, {dur_2/total_duration*100:.0f}%) — {time.time()-t0:.0f}s")
    results["pass_2"] = {"kept": keep_2, "raw_response": r2.choices[0].message.content}

    with open(os.path.join(OUTPUT_DIR, "c5296_pass2_raw.txt"), "w") as f:
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
        keep_3 = keep_2
    dur_3 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_3))
    print(f"  [Pass 3] {len(keep_3)}/{len(survivors_2)} lines kept ({dur_3:.0f}s, {dur_3/total_duration*100:.0f}%) — {time.time()-t0:.0f}s")
    results["pass_3"] = {"kept": keep_3, "raw_response": r3.choices[0].message.content}

    with open(os.path.join(OUTPUT_DIR, "c5296_pass3_raw.txt"), "w") as f:
        f.write(r3.choices[0].message.content)

    return results


def build_rlhf_file(lines, results, total_duration):
    """Build the annotatable RLHF review file."""
    keep_1 = set(results["pass_1"]["kept"])
    keep_2 = set(results["pass_2"]["kept"])
    keep_3 = set(results["pass_3"]["kept"])

    dur_1 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_1)
    dur_2 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_2)
    dur_3 = sum(l["end"] - l["start"] for l in lines if l["id"] in keep_3)

    out = []
    out.append("=" * 90)
    out.append("RLHF REVIEW — Project C5296 (32 min raw → 14:38 published)")
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
    out.append("    --> WRONG: should be KEEP, this is clean delivery")
    out.append("    --> WRONG: should be CUT, this is a false start")
    out.append("    --> WRONG: duplicate of L0234, keep that one instead")
    out.append("    --> OK")
    out.append("    --> this whole section (L0100-L0120) is a redo of L0050-L0070")
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

    path = os.path.join(OUTPUT_DIR, "c5296_rlhf_review.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"\n  RLHF review file: {path}")
    return path


def main():
    audio_path = step_1_extract_audio()
    words = step_2_transcribe(audio_path)
    total_duration = words[-1]["end"] if words else 0
    lines = step_3_build_lines(words)
    results = step_4_run_passes(lines, total_duration)
    build_rlhf_file(lines, results, total_duration)
    print("\nDone. Review the file, add your --> comments, then I'll rebuild the prompts.")


if __name__ == "__main__":
    main()
