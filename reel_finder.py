#!/usr/bin/env python3
"""Reel-finder: turn a long interview/testimonial transcript into ranked,
bite-sized reel clips with timestamps — for marketing.

This is the "find the gold" counterpart to the keep/cut pipeline. Instead of
deciding which takes are keepers, it scans the WHOLE transcript and surfaces
self-contained 8-60s moments that work as social-proof reels: specific
results/numbers, before->after transformation, emotional/authentic beats,
objection-handling, and crisp quotable one-liners.

Input  : word-level WhisperX json (outputs/rlhf/<tag>_words_whisperx.json)
Output : outputs/reel-finder/<tag>_reels.json   (structured clips)
         outputs/reel-finder/<tag>_reels.md     (human-readable, ranked)

Usage:
  python reel_finder.py david_ortiz                  # resolves the rlhf json by tag
  python reel_finder.py outputs/rlhf/foo_words_whisperx.json
  python reel_finder.py timm_freeman --min 10 --max 55 --max-clips 30
  python reel_finder.py josh_boucher --subject "Josh Boucher" --context "AAA Accelerator member"

Full coverage: the entire transcript goes to GPT-5.4 in one pass (1M context),
so nothing is truncated. The prompt forces an exhaustive scan of every section.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# reuse the pipeline's line-builder + formatter so line ids/timestamps match
from processor_v2 import _build_numbered_lines, _format_for_llm

HERE = Path(__file__).resolve().parent
# Output base: Danyal's workspace if present, else the repo dir (portable for any editor's clone).
_WS = Path("/Users/dany/Documents/Claude Workspaces/personal-workspace")
BASE = _WS if _WS.exists() else HERE
RLHF_DIR = BASE / "outputs/rlhf"
OUT_DIR = BASE / "outputs/reel-finder"
MODEL = "gpt-5.4"

load_dotenv(HERE / ".env")


def _tc(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600); s -= h * 3600
    m = int(s // 60); s -= m * 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


PROMPT = """You are a short-form video strategist who cuts marketing REELS from long testimonial / interview footage.

You will receive a numbered, timestamped transcript of ONE person being interviewed. Each line looks like:
  L0042 [123.4s-127.8s] the actual words spoken

{subject_block}{context_block}

YOUR JOB
Scan the ENTIRE transcript from the first line to the very last line and surface every self-contained moment that would make a strong standalone marketing reel. Be exhaustive — work section by section through the whole thing and do not stop early. It is better to return a borderline clip (low score) than to miss a good one.

WHAT MAKES A GREAT TESTIMONIAL REEL CLIP (look for all of these):
- SPECIFIC RESULTS / NUMBERS: revenue, clients signed, hours saved, % growth, time-to-result, money figures, headcount replaced.
- TRANSFORMATION (before -> after): "I used to ... now I ...", life/business change, identity shift.
- EMOTIONAL / AUTHENTIC BEATS: relief, disbelief, pride, gratitude, "this changed everything", vulnerability about the struggle before.
- OBJECTION HANDLING: skepticism that flipped — "I thought it was a scam / too good to be true / too expensive, but ...".
- QUOTABLE ONE-LINERS: punchy, confident, tweet-able statements about the program / mentorship / community / the work itself.
- AHA INSIGHTS / TACTICS: a concrete realization or method that gives a viewer real value.
- ENDORSEMENT: "best decision I made", "I'd tell anyone on the fence ...", direct recommendation.

CLIP RULES
- Length: each clip must be roughly {min_s}-{max_s} seconds of speech. Sweet spot 15-40s. Never below {min_s}s.
- SELF-CONTAINED: the clip must make sense on its own to a cold viewer. It must START at the beginning of a thought (not mid-sentence, not on a dangling "and/so/but/because/that" that refers to something earlier) and END on a completed thought.
- Set start_line to the first line of the thought and end_line to the last line of the thought. The clip spans those lines inclusive.
- quote MUST be the verbatim concatenation of the words in lines start_line..end_line (clean up only obvious transcription artifacts/spacing; do not paraphrase).
- Do not overlap clips heavily. If two strong moments are adjacent and flow together, you may make one longer clip (up to {max_s}s) or two separate clips — choose whatever cuts cleaner.
- Skip interviewer questions, logistics, dead chat, and rambling with no payoff.

FOR EACH CLIP RETURN:
- start_line, end_line   (integers, the L#### numbers without the "L")
- category               (one of: result, transformation, emotional, objection, quotable, insight, endorsement)
- hook                   (<= 8 words, punchy on-screen text caption that would make someone stop scrolling; in the speaker's authentic voice, NOT salesy hype)
- quote                  (verbatim transcript of the clip)
- score                  (integer 1-10: how strong/postable this is as a standalone reel)
- why                    (one sentence: why this works as a reel)

OUTPUT
Return ONLY valid JSON: an array of clip objects, sorted by score descending. No markdown, no commentary, no code fences."""


def _parse_clips(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    # grab the outermost array
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b != -1:
        text = text[a : b + 1]
    return json.loads(text)


def find_reels(words: list[dict], subject: str | None, context: str | None,
               min_s: float, max_s: float, max_clips: int) -> tuple[list[dict], list[dict]]:
    lines = _build_numbered_lines(words)          # ~15-word numbered lines w/ start/end
    by_id = {l["id"]: l for l in lines}
    transcript = _format_for_llm(lines)
    total = words[-1]["end"] if words else 0.0

    subject_block = f"\nThe person speaking is: {subject}.\n" if subject else ""
    context_block = f"Context: {context}.\n" if context else ""
    instructions = PROMPT.format(
        subject_block=subject_block, context_block=context_block,
        min_s=int(min_s), max_s=int(max_s),
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"  {len(lines)} lines / {len(words)} words / {total/60:.1f} min -> {MODEL}")
    r = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=transcript,
        reasoning={"effort": "high"},
        max_output_tokens=60000,
    )
    u = r.usage
    cost = u.input_tokens * 2.50 / 1_000_000 + u.output_tokens * 15.00 / 1_000_000
    print(f"  tokens in {u.input_tokens} / out {u.output_tokens} | ${cost:.4f}")

    raw = _parse_clips(r.output_text)
    clips = []
    for c in raw:
        try:
            sid, eid = int(c["start_line"]), int(c["end_line"])
        except (KeyError, ValueError, TypeError):
            continue
        if sid not in by_id or eid not in by_id or eid < sid:
            continue
        start = by_id[sid]["start"]
        end = by_id[eid]["end"]
        dur = end - start
        clips.append({
            "score": int(c.get("score", 0)),
            "category": c.get("category", "").strip(),
            "hook": c.get("hook", "").strip(),
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(dur, 2),
            "start_tc": _tc(start),
            "end_tc": _tc(end),
            "start_line": sid,
            "end_line": eid,
            "quote": " ".join(c.get("quote", "").split()),
            "why": c.get("why", "").strip(),
        })
    clips.sort(key=lambda x: (-x["score"], x["start"]))
    for i, c in enumerate(clips, 1):
        c["rank"] = i
    if max_clips:
        clips = clips[:max_clips]
    return clips, lines


def write_outputs(tag: str, clips: list[dict], subject: str | None, total_min: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / f"{tag}_reels.json"
    jpath.write_text(json.dumps(clips, indent=2))

    md = []
    md.append(f"# Reel clips — {subject or tag}")
    md.append("")
    md.append(f"Source transcript: `outputs/rlhf/{tag}_words_whisperx.json` ({total_min:.1f} min)")
    md.append(f"{len(clips)} clips found, ranked by reel-strength.")
    md.append("")
    md.append("`ranges` below plug straight into the XML cutter "
              "(`reels_from_ranges.py` style: `(start, end)` seconds).")
    md.append("")
    for c in clips:
        md.append(f"## #{c['rank']} · {c['score']}/10 · {c['category']}  — {c['hook']}")
        md.append(f"- **In/Out:** `{c['start_tc']} → {c['end_tc']}`  ({c['duration']:.1f}s)")
        md.append(f"- **Seconds:** `({c['start']}, {c['end']})`")
        md.append(f"- **Why:** {c['why']}")
        md.append(f"> {c['quote']}")
        md.append("")
    # convenient ranges dict
    md.append("---")
    md.append("### Ranges (paste into a REELS dict)")
    md.append("```python")
    md.append("REELS = {")
    for c in clips:
        key = f"r{c['rank']:02d}_{re.sub(r'[^a-z0-9]+','_', c['category'].lower())}"
        md.append(f'    "{key}": [({c["start"]}, {c["end"]})],   # {c["hook"]}')
    md.append("}")
    md.append("```")
    mpath = OUT_DIR / f"{tag}_reels.md"
    mpath.write_text("\n".join(md) + "\n")
    return jpath, mpath


def resolve_words(arg: str) -> tuple[Path, str]:
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        stem = p.stem
        tag = re.sub(r"_words_whisperx$", "", stem)
        return p, tag
    # treat as a tag
    cand = RLHF_DIR / f"{arg}_words_whisperx.json"
    if cand.exists():
        return cand, arg
    raise SystemExit(f"Could not find words json for '{arg}' (looked for {cand})")


def main():
    ap = argparse.ArgumentParser(description="Find marketing reel clips in a transcript.")
    ap.add_argument("input", help="tag (e.g. david_ortiz) or path to *_words_whisperx.json")
    ap.add_argument("--subject", default=None, help="who is speaking (e.g. 'David Ortiz')")
    ap.add_argument("--context", default=None, help="context (e.g. 'AAA Accelerator member testimonial')")
    ap.add_argument("--min", type=float, default=8.0, dest="min_s")
    ap.add_argument("--max", type=float, default=60.0, dest="max_s")
    ap.add_argument("--max-clips", type=int, default=30)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (expected in scripts/morningside-xml-pipeline/.env)")

    wpath, tag = resolve_words(args.input)
    words = json.loads(wpath.read_text())
    total_min = (words[-1]["end"] if words else 0) / 60.0
    print(f"Reel-finder: {wpath.name}  (tag={tag})")
    clips, _ = find_reels(words, args.subject, args.context, args.min_s, args.max_s, args.max_clips)
    jpath, mpath = write_outputs(tag, clips, args.subject, total_min)
    print(f"\n{len(clips)} clips → {jpath}")
    print(f"            → {mpath}")
    top = clips[:8]
    print("\nTop clips:")
    for c in top:
        print(f"  #{c['rank']} [{c['score']}/10] {c['start_tc']}-{c['end_tc']} ({c['duration']:.0f}s) {c['category']:13s} {c['hook']}")


if __name__ == "__main__":
    main()
