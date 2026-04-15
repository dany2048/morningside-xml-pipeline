#!/usr/bin/env python3
"""End-to-end pipeline test (now generalized — works on any raw MP4):

    WhisperX forced-aligned words  (cached at outputs/rlhf/<tag>_words_whisperx.json)
      -> processor_v2.process_lines(word_level=True) via GPT-5.4
         (or --keeps-file for manual / Claude picks)
      -> clean FCP7 XML with source-timecode element on <file>
         (fixes the XAVC start-TC bug)
      -> matching RLHF review txt

Usage:
    python test_c4109_e2e.py                                            # defaults to C4109
    python test_c4109_e2e.py --file path/to/raw.MP4                     # any clip
    python test_c4109_e2e.py --file ... --keeps-file keeps.json --tag v5-claude

Frame rate, source timecode, resolution, audio channels are all probed
from the source file, so 23.976 / 29.97 / 25 / 30 all work transparently.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
RLHF_DIR = WORKSPACE / "outputs/rlhf"
DEFAULT_SOURCE_MP4 = WORKSPACE / "reference/Raw Files Tests/20260203_C4109 (shorter clip) .MP4"


def _slugify(stem: str) -> str:
    """Make a filesystem-friendly tag from a file stem.
    'C4109 (shorter clip) ' -> 'c4109'; '20260208_C4340' -> 'c4340'.
    """
    m = re.search(r"[Cc]\d{3,5}", stem)
    if m:
        return m.group(0).lower()
    return re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()


# Frame-rate state populated by probe(). NOT module constants — they
# get rewritten per source file because different cameras shoot at
# different rates (C4109 is 23.976, C4340 is 29.97, etc).
_FPS_NUM = 24000
_FPS_DEN = 1001
_TIMEBASE = 24
_NTSC = True


def seconds_to_frames(s: float) -> int:
    return round(s * _FPS_NUM / _FPS_DEN)


def probe(path: Path) -> dict:
    """Run ffprobe and return the metadata dict + populate frame-rate state.

    Recognized rates: 23.976 (24000/1001), 24 (24/1), 25 (25/1),
    29.97 (30000/1001), 30 (30/1), 50, 59.94, 60. NTSC variants get
    timebase=ceil(rate) + ntsc=TRUE.
    """
    global _FPS_NUM, _FPS_DEN, _TIMEBASE, _NTSC

    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    # Frame rate
    fps_num, fps_den = (int(x) for x in v["r_frame_rate"].split("/"))
    fps = fps_num / fps_den
    # Map to FCP7 timebase + ntsc flag
    rate_map = {
        (24000, 1001): (24, True),    # 23.976
        (24, 1):       (24, False),   # 24
        (25, 1):       (25, False),   # 25
        (30000, 1001): (30, True),    # 29.97
        (30, 1):       (30, False),   # 30
        (50, 1):       (50, False),   # 50
        (60000, 1001): (60, True),    # 59.94
        (60, 1):       (60, False),   # 60
    }
    if (fps_num, fps_den) in rate_map:
        _TIMEBASE, _NTSC = rate_map[(fps_num, fps_den)]
        _FPS_NUM, _FPS_DEN = fps_num, fps_den
    else:
        # Fallback: use ffprobe's reported rate raw, ntsc off
        _FPS_NUM, _FPS_DEN = fps_num, fps_den
        _TIMEBASE = round(fps)
        _NTSC = False

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
        "fps": fps,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "timebase": _TIMEBASE,
        "ntsc": _NTSC,
    }


def tc_to_frames(tc: str, timebase: int) -> int:
    h, m, s, f = (int(x) for x in tc.replace(";", ":").split(":"))
    return (h * 3600 + m * 60 + s) * timebase + f


# ---------- XML generation ----------

def _rate(parent):
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(_TIMEBASE)
    ET.SubElement(r, "ntsc").text = "TRUE" if _NTSC else "FALSE"


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
    ET.SubElement(tc_el, "frame").text = str(tc_to_frames(start_tc, _TIMEBASE))
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


NEST_ID = "nest-1"


def _build_nest(parent: ET.Element, meta: dict, total_frames: int,
                filename: str, filepath: str, start_tc: str) -> ET.Element:
    """Build the inline NEST sequence — full source + empty mic track.

    The source <file> carries the source TC (preserving the C4109 fix);
    the nest sequence itself starts at 00:00:00:00, so Clean Cut clipitems
    can reference nest-frame offsets directly from seconds_to_frames().
    """
    audio_channels = meta["audio_channels"]
    nest = ET.SubElement(parent, "sequence", id=NEST_ID)
    ET.SubElement(nest, "name").text = f"NEST - {Path(filename).stem}"
    ET.SubElement(nest, "duration").text = str(total_frames)
    _rate(nest)

    ntc = ET.SubElement(nest, "timecode")
    _rate(ntc)
    ET.SubElement(ntc, "string").text = "00:00:00:00"
    ET.SubElement(ntc, "frame").text = "0"
    ET.SubElement(ntc, "displayformat").text = "NDF"

    n_media = ET.SubElement(nest, "media")

    # Video — full source
    n_video = ET.SubElement(n_media, "video")
    n_vfmt = ET.SubElement(n_video, "format")
    n_vsc = ET.SubElement(n_vfmt, "samplecharacteristics")
    _rate(n_vsc)
    ET.SubElement(n_vsc, "width").text = str(meta["width"])
    ET.SubElement(n_vsc, "height").text = str(meta["height"])
    ET.SubElement(n_vsc, "anamorphic").text = "FALSE"
    ET.SubElement(n_vsc, "pixelaspectratio").text = "square"
    ET.SubElement(n_vsc, "fielddominance").text = "none"

    n_vtrack = ET.SubElement(n_video, "track")
    nv_clip = ET.SubElement(n_vtrack, "clipitem", id="nest-v1")
    ET.SubElement(nv_clip, "name").text = filename
    ET.SubElement(nv_clip, "enabled").text = "TRUE"
    ET.SubElement(nv_clip, "duration").text = str(total_frames)
    _rate(nv_clip)
    ET.SubElement(nv_clip, "start").text = "0"
    ET.SubElement(nv_clip, "end").text = str(total_frames)
    ET.SubElement(nv_clip, "in").text = "0"
    ET.SubElement(nv_clip, "out").text = str(total_frames)
    _file_element(nv_clip, True, meta, total_frames, filename, filepath, start_tc)

    # Audio — camera channels (full clip) + empty external mic track
    n_audio = ET.SubElement(n_media, "audio")
    n_afmt = ET.SubElement(n_audio, "format")
    n_asc = ET.SubElement(n_afmt, "samplecharacteristics")
    ET.SubElement(n_asc, "depth").text = "16"
    ET.SubElement(n_asc, "samplerate").text = str(meta["sample_rate"])

    # Single audio track — source is effectively mono, ignore L/R split
    n_atrack = ET.SubElement(n_audio, "track")
    na_clip = ET.SubElement(n_atrack, "clipitem", id="nest-a1")
    ET.SubElement(na_clip, "name").text = filename
    ET.SubElement(na_clip, "enabled").text = "TRUE"
    ET.SubElement(na_clip, "duration").text = str(total_frames)
    _rate(na_clip)
    ET.SubElement(na_clip, "start").text = "0"
    ET.SubElement(na_clip, "end").text = str(total_frames)
    ET.SubElement(na_clip, "in").text = "0"
    ET.SubElement(na_clip, "out").text = str(total_frames)
    _file_element(na_clip, False, meta, total_frames, filename, filepath, start_tc)
    st = ET.SubElement(na_clip, "sourcetrack")
    ET.SubElement(st, "mediatype").text = "audio"
    ET.SubElement(st, "trackindex").text = "1"

    return nest


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
    # Single audio track in Clean Cut — mirrors the nest's single mono track
    a_tracks = [ET.SubElement(audio, "track")]

    nest_defined = False

    for i, (seg, (in_f, out_f, start_f, end_f)) in enumerate(zip(segments, frames)):
        clip_id = f"clipitem-{i+1}"

        v_clip = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(v_clip, "name").text = seg["label"]
        ET.SubElement(v_clip, "enabled").text = "TRUE"
        ET.SubElement(v_clip, "duration").text = str(total_frames)
        _rate(v_clip)
        ET.SubElement(v_clip, "start").text = str(start_f)
        ET.SubElement(v_clip, "end").text = str(end_f)
        ET.SubElement(v_clip, "in").text = str(in_f)
        ET.SubElement(v_clip, "out").text = str(out_f)

        if not nest_defined:
            _build_nest(v_clip, meta, total_frames, filename, filepath, start_tc)
            nest_defined = True
        else:
            ET.SubElement(v_clip, "sequence", id=NEST_ID)

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
            ET.SubElement(a_clip, "sequence", id=NEST_ID)
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
    out.append(f"Source: {meta.get('source_filename', '<unknown>')}")
    out.append(f"Source start TC: {meta['start_tc']} (frame {tc_to_frames(meta['start_tc'], _TIMEBASE)} at tb {_TIMEBASE})")
    out.append(f"FPS: {meta['fps']:.3f} ({meta['fps_num']}/{meta['fps_den']}, ntsc={meta['ntsc']})")
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
    parser = argparse.ArgumentParser(description="End-to-end test: WhisperX cache + LLM picks + FCP7 XML")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_SOURCE_MP4),
        help="Path to source MP4 (default: C4109 test clip)",
    )
    parser.add_argument(
        "--words-json",
        help="Path to WhisperX words JSON. Defaults to outputs/rlhf/<tag>_words_whisperx.json",
    )
    parser.add_argument(
        "--keeps-file",
        help="Path to JSON file containing a flat array of line IDs to keep. "
             "If set, skips the GPT-5.4 LLM call and uses these picks directly.",
    )
    parser.add_argument(
        "--tag",
        default="v4",
        help="Output tag (default 'v4' for GPT-5.4 run; e.g. 'v5-claude' for manual keeps).",
    )
    args = parser.parse_args()

    source_mp4 = Path(args.file).expanduser().resolve()
    if not source_mp4.exists():
        raise SystemExit(f"File not found: {source_mp4}")
    clip_tag = _slugify(source_mp4.stem)

    words_json = Path(args.words_json).resolve() if args.words_json else (
        RLHF_DIR / f"{clip_tag}_words_whisperx.json"
    )
    if not words_json.exists():
        raise SystemExit(
            f"WhisperX words file not found: {words_json}\n"
            f"Run: python run_whisperx_c4109.py --file '{source_mp4}'"
        )

    out_xml = RLHF_DIR / f"{clip_tag}_{args.tag}_cut.xml"
    out_review = RLHF_DIR / f"{clip_tag}_{args.tag}_review.txt"

    t0 = time.time()

    print(f"Source: {source_mp4}")
    print(f"Clip tag: {clip_tag}")
    meta = probe(source_mp4)
    print(f"  duration: {meta['duration']:.2f}s  |  {meta['width']}x{meta['height']}  |  "
          f"fps: {meta['fps']:.3f}  |  start TC: {meta['start_tc']}  |  "
          f"timebase: {meta['timebase']} ntsc={meta['ntsc']}")

    words = json.loads(words_json.read_text())
    print(f"WhisperX words: {len(words)} (from {words_json.name})")
    meta["source_filename"] = source_mp4.name

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
    build_xml(segments, meta, source_mp4.name, str(source_mp4), out_xml)
    print(f"  → {out_xml}")

    print(f"\n[4] Writing RLHF review...")
    write_review(words, lines, keep_ids_set, segments, meta, out_review)
    print(f"  → {out_review}")

    total_kept = sum(s["end"] - s["start"] for s in segments)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Kept: {total_kept:.1f}s / {meta['duration']:.1f}s ({total_kept/meta['duration']*100:.1f}%)")


if __name__ == "__main__":
    main()
