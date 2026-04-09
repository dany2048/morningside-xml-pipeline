"""Core processing: Line-numbered transcript + GPT-4o multi-pass analysis."""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from config import REMOVABLE_PAUSE, SEGMENT_PADDING_SECONDS


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

    # Merge consecutive/close lines into segments
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


PASS_1_PROMPT = """You are a professional video editor analyzing raw talking-head footage.

Below is a numbered transcript from a raw single-camera recording. The creator speaks to camera about AI topics. The raw footage contains:

- GOOD TAKES: Polished, on-topic, to-camera delivery (KEEP these)
- REPEATED TAKES: Same section re-done with better wording, sometimes minutes apart (keep ONLY the last/best complete version)
- MICRO-JITTERS: False starts like "and in. and in, and in this video..." (keep ONLY the final clean phrase)
- OFF-CAMERA TALK: "I'm filming can we keep it down", "I'm locked in today bro", chatter with people nearby (CUT all)
- MUMBLING / UNCLEAR: Background noise, half-words, non-content (CUT all)
- SELF-DIRECTION: "Let me try that again", "okay", throat clearing (CUT all)

YOUR TASK: Return ONLY the line numbers that should be KEPT in the final edit.

Rules:
- Be aggressive. Only keep clean, complete, on-topic delivery.
- When the creator says the same thing multiple times, keep ONLY the last complete version.
- For false starts / jitters within a line, keep the line ONLY if the final phrase in it is usable — the video editor can trim within the line later.
- If in doubt, CUT. An editor can always add back, but a bloated rough cut is useless.
- A typical 30-min raw recording produces a 10-15 min final video. If you're keeping more than 50%, you're not cutting enough.

Respond with ONLY a JSON array of line numbers to keep, e.g.:
[3, 4, 5, 12, 13, 14, 28, 29, 30]

No explanation. Just the array."""


PASS_2_PROMPT = """You are a senior editor doing a CRITICAL second pass on a rough cut.

Below is the transcript that survived the first cut. Read it TOP TO BOTTOM as a viewer would experience it.

Your #1 priority is FALSE STARTS and MICRO-JITTERS. These are the most common problem that survives a first pass:
- "And in. And in, and in this video" → only the final complete phrase should survive
- "So what we're going to. So what we're going to do is" → cut the incomplete attempt
- "The reason, the reason why, the real reason why this works" → keep only the last version
- A line that begins mid-thought because the previous attempt was cut → cut this fragment too
- "So..." or "Right..." or "Okay so..." dangling at the start of a section → cut it

ALSO check for:
- Repeated ideas: same point made twice in different words → keep the better one
- Incomplete thoughts: lines that trail off or don't land → cut
- Off-topic: any remaining self-talk, side comments → cut
- Orphaned connectors: "and" or "but" or "so" at the start of a line that no longer connects to anything → cut

Read the whole thing. Then ask: "If I read this aloud, would it sound like a polished, rehearsed script?" If any line sounds like a stumble, CUT IT.

Return ONLY a JSON array of line numbers to KEEP. Nothing else."""


PASS_3_PROMPT = """FINAL CHECK. Read this transcript as if you are about to record it as a voiceover.

Read every single line out loud in your head. Does it flow? Is every sentence complete? Is every transition smooth?

Kill anything that:
- Starts with a stutter or restart ("I, I think" / "what, what we need")
- Repeats a point already made earlier (even if worded differently)
- Is a fragment that doesn't form a complete thought
- Starts with a dangling connector that doesn't connect to anything
- Is filler disguised as content ("so basically" / "right so" / "okay now")

Be aggressive on this final pass. Every single line must earn its place. If you're unsure about a line, CUT IT. An editor can always restore — but a bloated cut wastes everyone's time.

Return ONLY a JSON array of the final line numbers to keep. Nothing else."""


def process(words: list[dict], total_duration: float) -> list[dict]:
    """Multi-pass GPT-4o processing with numbered lines.

    Pass 1: Full transcript → aggressive cut
    Pass 2: Survivors only → catch semantic repeats, incomplete thoughts
    Pass 3: Final QC → perfection check

    Returns list of {start: float, end: float, label: str} segments.
    """
    client = _get_client()

    # Build numbered lines
    lines = _build_numbered_lines(words)
    print(f"  {len(lines)} numbered lines from {len(words)} words")

    full_transcript = _format_for_llm(lines)

    # === PASS 1 ===
    print("\n  [Pass 1] Full transcript → GPT-4o...")
    r1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_1_PROMPT},
            {"role": "user", "content": full_transcript},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    print(f"  [Pass 1] Tokens — in: {r1.usage.prompt_tokens}, out: {r1.usage.completion_tokens}")
    keep_1 = _parse_line_numbers(r1.choices[0].message.content)
    # Validate line IDs exist
    valid_ids = {l["id"] for l in lines}
    keep_1 = [k for k in keep_1 if k in valid_ids]
    kept_dur_1 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_1))
    print(f"  [Pass 1] Keeping {len(keep_1)}/{len(lines)} lines ({kept_dur_1:.0f}s, {kept_dur_1/total_duration*100:.0f}%)")

    if not keep_1:
        raise RuntimeError("Pass 1 returned no lines")

    # === PASS 2 ===
    survivors_1 = [l for l in lines if l["id"] in set(keep_1)]
    survivors_1_text = _format_for_llm(survivors_1)

    print("\n  [Pass 2] Survivors → GPT-4o...")
    r2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_2_PROMPT},
            {"role": "user", "content": survivors_1_text},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    print(f"  [Pass 2] Tokens — in: {r2.usage.prompt_tokens}, out: {r2.usage.completion_tokens}")
    keep_2 = _parse_line_numbers(r2.choices[0].message.content)
    keep_2 = [k for k in keep_2 if k in valid_ids]
    kept_dur_2 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_2))
    print(f"  [Pass 2] Keeping {len(keep_2)}/{len(survivors_1)} lines ({kept_dur_2:.0f}s, {kept_dur_2/total_duration*100:.0f}%)")

    if not keep_2:
        print("  Warning: Pass 2 empty, using Pass 1")
        keep_2 = keep_1

    # === PASS 3 ===
    survivors_2 = [l for l in lines if l["id"] in set(keep_2)]
    survivors_2_text = _format_for_llm(survivors_2)

    print("\n  [Pass 3] Final QC → GPT-4o...")
    r3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PASS_3_PROMPT},
            {"role": "user", "content": survivors_2_text},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    print(f"  [Pass 3] Tokens — in: {r3.usage.prompt_tokens}, out: {r3.usage.completion_tokens}")
    keep_3 = _parse_line_numbers(r3.choices[0].message.content)
    keep_3 = [k for k in keep_3 if k in valid_ids]
    kept_dur_3 = sum(l["end"] - l["start"] for l in lines if l["id"] in set(keep_3))
    print(f"  [Pass 3] Keeping {len(keep_3)}/{len(survivors_2)} lines ({kept_dur_3:.0f}s, {kept_dur_3/total_duration*100:.0f}%)")

    if not keep_3:
        print("  Warning: Pass 3 empty, using Pass 2")
        keep_3 = keep_2

    # Convert to time segments
    raw_segments = _lines_to_segments(lines, keep_3)

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

    return segments
