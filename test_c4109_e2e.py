#!/usr/bin/env python3
"""End-to-end test on C4109 with the full fix stack:

    WhisperX forced-aligned words  (outputs/rlhf/c4109_words_whisperx.json)
      -> processor_v2.process_lines(word_level=True) via GPT-5.4
      -> clean FCP7 XML with source-timecode element (fixes the 03:35:01:00 bug)
      -> matching RLHF review txt

Outputs:
    outputs/rlhf/c4109_v4_cut.xml
    outputs/rlhf/c4109_v4_review.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
from xml.dom import minidom

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent.parent / ".env")

from config import SEGMENT_PADDING_SECONDS
from processor_v2 import _build_numbered_lines, _lines_to_segments, process_lines

WORKSPACE = Path("/Users/dany/Documents/Claude Workspaces/personal-workspace")
WORDS_JSON = WORKSPACE / "outputs/rlhf/c4109_words_whisperx.json"
SOURCE_MP4 = WORKSPACE / "reference/Raw Files Tests/20260203_C4109 (shorter clip) .MP4"

FPS_NUM = 24000
FPS_DEN = 1001
TIMEBASE = 24
NTSC = True


def seconds_to_frames(s: float) -> int:
    return round(s * FPS_NUM / FPS_DEN)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    start_tc = "00:00:00:00"
    for s in data.get("streams", []):
        tc = s.get("tags", {}).get("timecode")
        if tc and ":" in tc:
            start_tc = tc[:11]
            break
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "sample_rate": int(a["sample_rate"]) if a else 48000,
        "audio_channels": int(a.get("channels", 2)) if a else 2,
        "start_tc": start_tc,
    }


def tc_to_frames(tc: str, timebase: int) -> int:
    h, m, s, f = (int(x) for x in tc.replace(";", ":").split(":"))
    return (h * 3600 + m * 60 + s) * timebase + f


# ---------- XML generation ----------

def _rate(parent):
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(TIMEBASE)
    ET.SubElement(r, "ntsc").text = "TRUE" if NTSC else "FALSE"


def _file_element(parent, define: bool, meta: dict, total_frames: int,
                  filename: str, filepath: str, start_tc: str):
    f = ET.SubElement(parent, "file", id="file-1")
    if not define:
        return f
    ET.SubElement(f, "name").text = filename
    ET.SubElement(f, "pathurl").text = "file://localhost" + quote(filepath, safe="/")
    _rate(f)
    ET.SubElement(f, "duration").text = str(total_frames)

    tc_el = ET.SubElement(f, "timecode")
    _rate(tc_el)
    ET.SubElement(tc_el, "string").text = start_tc
    ET.SubElement(tc_el, "frame").text = str(tc_to_frames(start_tc, TIMEBASE))
    ET.SubElement(tc_el, "displayformat").text = "NDF"
    ET.SubElement(tc_el, "source").text = "source"

    media = ET.SubElement(f, "media")
    v = ET.SubElement(media, "video")
    vsc = ET.SubElement(v, "samplecharacteristics")
    _rate(vsc)
    ET.SubElement(vsc, "width").text = str(meta["width"])
    ET.SubElement(vsc, "height").text = str(meta["height"])
    ET.SubElement(vsc, "anamorphic").text = "FALSE"
    ET.SubElement(vsc, "pixelaspectratio").text = "square"
    ET.SubElement(vsc, "fielddominance").text = "none"
    a = ET.SubElement(media, "audio")
    ET.SubElement(a, "channelcount").text = str(meta["audio_channels"])
    asc = ET.SubElement(a, "samplecharacteristics")
    ET.SubElement(asc, "depth").text = "16"
    ET.SubElement(asc, "samplerate").text = str(meta["sample_rate"])
    return f


def build_xml(segments: list[dict], meta: dict, filename: str, filepath: str, out_path: Path) -> None:
    total_frames = seconds_to_frames(meta["duration"])
    start_tc = meta["start_tc"]
    audio_channels = meta["audio_channels"]

    # Pre-compute per-segment frames.
    frames = []
    cum = 0
    for s in segments:
        in_f = seconds_to_frames(s["start"])
        out_f = seconds_to_frames(s["end"])
        dur = out_f - in_f
        frames.append((in_f, out_f, cum, cum + dur))
        cum += dur
    timeline_total = cum

    xmeml = ET.Element("xmeml", version="5")
    seq = ET.SubElement(xmeml, "sequence", id="sequence-1")
    ET.SubElement(seq, "name").text = f"{Path(filename).stem} - Clean Cut v4"
    ET.SubElement(seq, "duration").text = str(timeline_total)
    _rate(seq)

    tc = ET.SubElement(seq, "timecode")
    _rate(tc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"
    ET.SubElement(tc, "source").text = "source"

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    vfmt = ET.SubElement(video, "format")
    vsc = ET.SubElement(vfmt, "samplecharacteristics")
    _rate(vsc)
    ET.SubElement(vsc, "width").text = str(meta["width"])
    ET.SubElement(vsc, "height").text = str(meta["height"])
    ET.SubElement(vsc, "anamorphic").text = "FALSE"
    ET.SubElement(vsc, "pixelaspectratio").text = "square"
    ET.SubElement(vsc, "fielddominance").text = "none"
    v_track = ET.SubElement(video, "track")

    audio = ET.SubElement(media, "audio")
    afmt = ET.SubElement(audio, "format")
    asc = ET.SubElement(afmt, "samplecharacteristics")
    ET.SubElement(asc, "depth").text = "16"
    ET.SubElement(asc, "samplerate").text = str(meta["sample_rate"])
    a_tracks = [ET.SubElement(audio, "track") for _ in range(audio_channels)]

    for i, (seg, (in_f, out_f, start_f, end_f)) in enumerate(zip(segments, frames)):
        clip_id = f"clipitem-{i+1}"
        define_file = (i == 0)

        v_clip = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(v_clip, "name").text = seg["label"]
        ET.SubElement(v_clip, "enabled").text = "TRUE"
        ET.SubElement(v_clip, "duration").text = str(total_frames)
        _rate(v_clip)
        ET.SubElement(v_clip, "start").text = str(start_f)
        ET.SubElement(v_clip, "end").text = str(end_f)
        ET.SubElement(v_clip, "in").text = str(in_f)
        ET.SubElement(v_clip, "out").text = str(out_f)
        _file_element(v_clip, define_file, meta, total_frames, filename, filepath, start_tc)

        for ch, atr in enumerate(a_tracks):
            a_clip = ET.SubElement(atr, "clipitem", id=f"{clip_id}-a{ch+1}")
            ET.SubElement(a_clip, "name").text = seg["label"]
            ET.SubElement(a_clip, "enabled").text = "TRUE"
            ET.SubElement(a_clip, "duration").text = str(total_frames)
            _rate(a_clip)
            ET.SubElement(a_clip, "start").text = str(start_f)
            ET.SubElement(a_clip, "end").text = str(end_f)
            ET.SubElement(a_clip, "in").text = str(in_f)
            ET.SubElement(a_clip, "out").text = str(out_f)
            _file_element(a_clip, False, meta, total_frames, filename, filepath, start_tc)
            st = ET.SubElement(a_clip, "sourcetrack")
            ET.SubElement(st, "mediatype").text = "audio"
            ET.SubElement(st, "trackindex").text = str(ch + 1)

    rough = ET.tostring(xmeml, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    body = "\n".join(pretty.split("\n")[1:])
    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body
    )


# ---------- Review ----------

def write_review(words: list[dict], lines: list[dict], keep_ids: set[int],
                 segments: list[dict], meta: dict, out_path: Path) -> None:
    out = []
    bar = "=" * 90
    out.append(bar)
    out.append("RLHF REVIEW V4 — C4109 — WhisperX forced-aligned + GPT-5.4 picks + v3 XML")
    out.append(bar)
    out.append("")
    out.append(f"Source: {SOURCE_MP4.name}")
    out.append(f"Source start TC: {meta['start_tc']} (frame {tc_to_frames(meta['start_tc'], TIMEBASE)} at tb {TIMEBASE})")
    out.append(f"Duration: {meta['duration']:.2f}s")
    out.append(f"Words (WhisperX): {len(words)}")
    out.append(f"Lines: {len(lines)}  |  Kept: {len(keep_ids)}")
    out.append(f"Segments: {len(segments)}")
    total = sum(s["end"] - s["start"] for s in segments)
    out.append(f"Kept duration: {total:.2f}s ({total/meta['duration']*100:.1f}%)")
    out.append("")
    out.append("SEGMENTS (= clipitems in XML):")
    cum = 0
    for i, s in enumerate(segments, 1):
        in_f = seconds_to_frames(s["start"])
        out_f = seconds_to_frames(s["end"])
        dur_f = out_f - in_f
        out.append(
            f"  {i:02d}  src {s['start']:7.3f}s→{s['end']:7.3f}s  "
            f"({s['end']-s['start']:5.2f}s)  in={in_f:6d} out={out_f:6d}  "
            f"timeline {cum:6d}→{cum+dur_f:6d}"
        )
        cum += dur_f
    out.append("")
    out.append(bar)
    out.append("WORD-BY-WORD (WhisperX) with KEEP/CUT from GPT-5.4 line-level picks")
    out.append(bar)

    # Flatten: map each word to KEEP/CUT based on which line it's in.
    word_to_line = {}
    for line in lines:
        # Each line has id, start, end, text — text is space-joined words.
        # Match by time window: any word whose center falls inside the line.
        for i, w in enumerate(words):
            mid = (w["start"] + w["end"]) / 2
            if line["start"] <= mid <= line["end"]:
                word_to_line[i] = line["id"]

    prev_end = 0.0
    for i, w in enumerate(words):
        gap = w["start"] - prev_end
        if gap > 0.5 and i > 0:
            out.append(f"  --- gap {gap:.2f}s ---")
        line_id = word_to_line.get(i)
        decision = "KEEP" if line_id in keep_ids else "CUT "
        out.append(f"W{i+1:04d} [{w['start']:7.2f}-{w['end']:7.2f}] {decision} {w['word']}")
        prev_end = w["end"]
    out_path.write_text("\n".join(out) + "\n")


# ---------- Main ----------

def segments_from_keeps(lines: list[dict], keep_ids: list[int], total_duration: float) -> list[dict]:
    """Same logic as processor_v2.process_lines() after the LLM call — convert
    a keep_ids list into padded segments."""
    raw = _lines_to_segments(lines, keep_ids)
    segments = []
    for seg in raw:
        segments.append({
            "start": max(0, seg["start"] - SEGMENT_PADDING_SECONDS),
            "end":   min(total_duration, seg["end"] + SEGMENT_PADDING_SECONDS),
            "label": f"seg_{len(segments) + 1}",
        })
    return segments


def main():
    parser = argparse.ArgumentParser(description="C4109 end-to-end test (WhisperX + LLM picks + FCP7 XML)")
    parser.add_argument(
        "--keeps-file",
        help="Path to JSON file containing a flat array of line IDs to keep. "
             "If set, skips the GPT-5.4 LLM call and uses these picks directly. "
             "Useful for comparing Claude Code picks vs GPT-5.4 on the same transcript.",
    )
    parser.add_argument(
        "--tag",
        default="v4",
        help="Output tag (default 'v4' for GPT-5.4 run; use e.g. 'v5-claude' for manual keeps).",
    )
    args = parser.parse_args()

    out_xml = WORKSPACE / f"outputs/rlhf/c4109_{args.tag}_cut.xml"
    out_review = WORKSPACE / f"outputs/rlhf/c4109_{args.tag}_review.txt"

    t0 = time.time()

    print(f"Source: {SOURCE_MP4}")
    meta = probe(SOURCE_MP4)
    print(f"  duration: {meta['duration']:.2f}s  |  {meta['width']}x{meta['height']}  |  start TC: {meta['start_tc']}")

    words = json.loads(WORDS_JSON.read_text())
    print(f"WhisperX words: {len(words)}")

    print("\n[1] Building word-level numbered lines...")
    lines = _build_numbered_lines(words, word_level=True)
    print(f"  {len(lines)} numbered lines (one per word, capped at start+1.0s)")

    if args.keeps_file:
        print(f"\n[2] Loading keep IDs from {args.keeps_file}...")
        keeps_path = Path(args.keeps_file)
        if not keeps_path.is_absolute():
            keeps_path = WORKSPACE / keeps_path
        keep_ids = json.loads(keeps_path.read_text())
        valid_line_ids = {l["id"] for l in lines}
        keep_ids = [k for k in keep_ids if k in valid_line_ids]
        kept_dur = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_ids))
        print(f"  Keeping {len(keep_ids)}/{len(lines)} lines ({kept_dur:.0f}s, "
              f"{kept_dur/meta['duration']*100:.0f}%) — manual picks (no LLM call)")
        segments = segments_from_keeps(lines, keep_ids, meta["duration"])
        print(f"  → {len(segments)} segments after adjacency merging")
    else:
        print("\n[2] GPT-5.4 keep/cut pass...")
        segments = process_lines(lines, meta["duration"])

    if not segments:
        print("ERROR: no segments produced")
        sys.exit(1)

    # Reconstruct kept line IDs for the review output (whether from keeps-file
    # or from GPT-5.4 segments)
    keep_ids_set = set()
    for line in lines:
        mid = (line["start"] + line["end"]) / 2
        for seg in segments:
            if seg["start"] <= mid <= seg["end"]:
                keep_ids_set.add(line["id"])
                break

    print(f"\n[3] Generating XML ({len(segments)} segments)...")
    build_xml(segments, meta, SOURCE_MP4.name, str(SOURCE_MP4), out_xml)
    print(f"  → {out_xml}")

    print(f"\n[4] Writing RLHF review...")
    write_review(words, lines, keep_ids_set, segments, meta, out_review)
    print(f"  → {out_review}")

    total_kept = sum(s["end"] - s["start"] for s in segments)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Kept: {total_kept:.1f}s / {meta['duration']:.1f}s ({total_kept/meta['duration']*100:.1f}%)")


if __name__ == "__main__":
    main()
