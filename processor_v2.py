"""Core processing: GPT-5.4 multi-pass analysis with 1M context window.

Replaces processor.py (GPT-4o). Same interface, better model.
GPT-5.4: $2.50/1M input, $15.00/1M output, 1M context, 128k max output.
"""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from config import REMOVABLE_PAUSE, SEGMENT_PADDING_SECONDS


MODEL = "gpt-5.4"


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _build_numbered_lines(words: list[dict]) -> list[dict]:
    """Break words into numbered lines. Each line has an ID, start, end, and text.

    Lines break on pauses > 0.3s or every ~15 words.
    """
    lines = []
    current_words = []
    current_start = None
    last_end = 0

    def flush():
        if current_words:
            text = " ".join(w["word"] for w in current_words)
            lines.append({
                "id": len(lines) + 1,
                "start": current_start,
                "end": current_words[-1]["end"],
                "text": text,
            })

    for w in words:
        if current_start is None:
            current_start = w["start"]

        gap = w["start"] - last_end if last_end > 0 else 0
        if gap > 0.3 and current_words:
            flush()
            current_words = []
            current_start = w["start"]

        current_words.append(w)
        last_end = w["end"]

        if len(current_words) >= 15:
            flush()
            current_words = []
            current_start = None

    flush()
    return lines


def _format_for_llm(lines: list[dict]) -> str:
    """Format numbered lines for LLM consumption."""
    return "\n".join(
        f"L{line['id']:04d} [{line['start']:.1f}s-{line['end']:.1f}s] {line['text']}"
        for line in lines
    )


def _parse_line_numbers(content: str) -> list[int]:
    """Parse line numbers from LLM response. Handles various formats."""
    # Remove markdown fences
    content = re.sub(r"```(?:json)?\s*", "", content).rstrip("`").strip()

    # Try JSON array first
    try:
        nums = json.loads(content)
        if isinstance(nums, list):
            return [int(n) for n in nums]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract all numbers that look like line references
    nums = re.findall(r"L?0*(\d+)", content)
    return [int(n) for n in nums]


def _lines_to_segments(lines: list[dict], keep_ids: list[int]) -> list[dict]:
    """Convert kept line IDs back to time segments, merging adjacent lines."""
    keep_set = set(keep_ids)
    kept_lines = [l for l in lines if l["id"] in keep_set]

    if not kept_lines:
        return []

    segments = []
    seg_start = kept_lines[0]["start"]
    seg_end = kept_lines[0]["end"]

    for line in kept_lines[1:]:
        gap = line["start"] - seg_end
        if gap <= REMOVABLE_PAUSE:
            seg_end = line["end"]
        else:
            segments.append({"start": seg_start, "end": seg_end})
            seg_start = line["start"]
            seg_end = line["end"]

    segments.append({"start": seg_start, "end": seg_end})
    return segments


def _cost_estimate(usage, label: str) -> float:
    """Calculate and print cost for a GPT-5.4 call."""
    input_cost = usage.input_tokens * 2.50 / 1_000_000
    output_cost = usage.output_tokens * 15.00 / 1_000_000
    total = input_cost + output_cost
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    print(f"  [{label}] Tokens — in: {usage.input_tokens}, out: {usage.output_tokens} (reasoning: {reasoning}) | Cost: ${total:.4f}")
    return total


SINGLE_PASS_PROMPT = """You are an expert video editor cutting raw talking-head footage for a YouTube channel about AI.

Below is a numbered, timestamped transcript from a single continuous recording. The creator films in one long take, repeating lines until he's happy with the delivery.

THE RAW FOOTAGE CONTAINS:
- KEEPER TAKES: Clean, confident, on-topic delivery. These go in the final video.
- REPEATED TAKES: The same sentence or paragraph said 2-8 times. Keep ONLY the last complete, fluent version. Earlier attempts are always worse — he practices delivery with each attempt and moves on once satisfied.
- FALSE STARTS: "And in. And in, and in this video..." — stuttering into a line. Keep ONLY the final clean run.
- MICRO-RESTARTS: Mid-sentence corrections. "The way that, the way this works is..." — keep only the completed version.
- OFF-CAMERA TALK: Talking to someone in the room, phone calls, "let me try that again", "hold on", laughing at mistakes. CUT all of it.
- SELF-DIRECTION: "Okay", "right", "let's go", throat clearing, sighing. CUT all.
- DEAD AIR / MUMBLING: Silence, background noise, half-words, garbled speech. CUT all.

HOW TO IDENTIFY REPEATED TAKES:
The creator records linearly. When you see the same idea expressed multiple times within a short window (usually 30-90 seconds), those are repeated takes. The LAST complete version is almost always the best. "Complete" means: ends with a full sentence, doesn't trail off, contains the full thought.

HOW TO IDENTIFY FALSE STARTS:
"And in. And in, and in this video..." — stuttering into a line. Keep ONLY the final clean run. The speaker saying the same 1-5 words repeatedly in a loop, keep ONLY the last take.

CRITICAL RULES:
- Do NOT over-cut. When a line is clean, on-topic delivery — KEEP IT even if it sounds slightly imperfect. The human editor will handle fine cuts.
- Only cut lines you are CONFIDENT are junk, repeats, or off-camera talk.
- When you see repeated takes, keep the LAST complete one, cut all earlier attempts.
- Connectors ("so", "and", "now") at the start of a line are fine if the line itself is good content — don't cut just because it starts with "so".

STRUCTURAL RULES (these catch errors that local line-by-line review misses):

1. TAKE GROUPING — BEFORE selecting which lines to keep, first divide the transcript into SECTIONS (intro/hook, main argument, feature comparison, CTA, outro, etc). Within each section, identify where the creator re-attempts the same passage. A re-attempt starts when you see the creator restart a thought he already tried earlier — similar opening words, same topic, same structure. Once you identify take groups, ONLY keep lines from the LAST attempt of each section. Do NOT mix lines from an early attempt with lines from a later attempt of the same section, even if the early attempt has a nice line that the later one doesn't. The later attempt is the creator's chosen version.

2. ORPHAN CHECK — After deciding your keep list, do a SECOND scan: read ONLY the kept lines in sequence. For each one, ask: "Does this line make grammatical and logical sense on its own, without the line before it?" If a kept line starts mid-sentence, references something not established, or is clearly the tail of a longer thought whose beginning was cut — REMOVE IT.

3. VISUAL REFERENCES — Lines where the creator says "this graph here", "as you can see", "let me draw this", "goes like this vs like this" are referencing on-screen visuals. These lines carry MORE value than the transcript suggests because the viewer sees the visual. Bias toward KEEP for these, even if the delivery has moderate stammering.

Respond with ONLY a JSON array of line numbers to KEEP.
No explanation. No commentary. Just the JSON array."""


def process_lines(lines: list[dict], total_duration: float) -> list[dict]:
    """Single-pass GPT-5.4 processing with pre-built numbered lines.

    Accepts lines from any source (Whisper word-level or Premiere transcript).
    Each line must have: id, start, end, text.

    Returns list of {start: float, end: float, label: str} segments.
    """
    client = _get_client()

    print(f"  {len(lines)} lines, {total_duration:.0f}s ({total_duration/60:.1f} min)")

    full_transcript = _format_for_llm(lines)
    valid_ids = {l["id"] for l in lines}

    # === SINGLE PASS ===
    print(f"\n  Full transcript -> {MODEL} (reasoning: medium)...")
    r = client.responses.create(
        model=MODEL,
        instructions=SINGLE_PASS_PROMPT,
        input=full_transcript,
        reasoning={"effort": "medium"},
        max_output_tokens=32000,
    )
    cost = _cost_estimate(r.usage, "Single pass")

    keep = _parse_line_numbers(r.output_text)
    keep = [k for k in keep if k in valid_ids]
    kept_dur = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep))
    print(f"  Keeping {len(keep)}/{len(lines)} lines ({kept_dur:.0f}s, {kept_dur/total_duration*100:.0f}%)")

    if not keep:
        raise RuntimeError("GPT-5.4 returned no lines to keep")

    # Convert to time segments
    raw_segments = _lines_to_segments(lines, keep)

    # Apply padding and label
    segments = []
    for seg in raw_segments:
        segments.append({
            "start": max(0, seg["start"] - SEGMENT_PADDING_SECONDS),
            "end": min(total_duration, seg["end"] + SEGMENT_PADDING_SECONDS),
            "label": f"seg_{len(segments) + 1}",
        })

    total_kept = sum(s["end"] - s["start"] for s in segments)
    print(f"\n  Final: {len(segments)} segments, {total_kept:.0f}s kept ({total_kept/total_duration*100:.0f}%) from {total_duration:.0f}s raw")
    print(f"  GPT-5.4 cost: ${cost:.4f}")

    return segments


def process(words: list[dict], total_duration: float) -> list[dict]:
    """Process from Whisper word-level data. Builds numbered lines then runs GPT-5.4."""
    lines = _build_numbered_lines(words)
    print(f"  Built {len(lines)} numbered lines from {len(words)} words")
    return process_lines(lines, total_duration)
