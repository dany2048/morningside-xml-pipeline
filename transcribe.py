"""Whisper transcription — OpenAI API (default) or local model (--local flag)."""
from __future__ import annotations

import os

from openai import OpenAI

from config import CHUNK_OVERLAP_SECONDS, WHISPER_MODEL


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


def transcribe_chunk_api(chunk_path: str, offset_seconds: float) -> list[dict]:
    """Transcribe via OpenAI Whisper API. Returns list of {word, start, end}."""
    client = _get_client()
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = []
    for w in getattr(result, "words", []) or []:
        words.append({
            "word": w.word.strip() if hasattr(w, "word") else w["word"].strip(),
            "start": (w.start if hasattr(w, "start") else w["start"]) + offset_seconds,
            "end": (w.end if hasattr(w, "end") else w["end"]) + offset_seconds,
        })

    return words


def transcribe_chunk_local(chunk_path: str, offset_seconds: float) -> list[dict]:
    """Transcribe via local openai-whisper model. Returns list of {word, start, end}."""
    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(chunk_path, word_timestamps=True, language="en")

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"] + offset_seconds,
                "end": w["end"] + offset_seconds,
            })
    return words


def transcribe_all(chunks: list[tuple[str, float]], use_local: bool = False) -> list[dict]:
    """Transcribe all chunks and merge, deduplicating overlap regions."""
    all_words = []
    transcribe_fn = transcribe_chunk_local if use_local else transcribe_chunk_api

    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        print(f"  Transcribing chunk {i+1}/{len(chunks)} ({chunk_size_mb:.1f} MB)...")
        words = transcribe_fn(chunk_path, offset)

        if i == 0 or not all_words:
            all_words.extend(words)
            continue

        # Deduplicate overlap region
        overlap_start = offset
        overlap_end = offset + CHUNK_OVERLAP_SECONDS

        new_words = []
        for w in words:
            if w["start"] >= overlap_end:
                new_words.append(w)
            elif w["start"] >= overlap_start:
                is_dup = False
                for existing in all_words[-20:]:
                    if (existing["word"].lower() == w["word"].lower()
                            and abs(existing["start"] - w["start"]) < 0.5):
                        is_dup = True
                        break
                if not is_dup:
                    new_words.append(w)

        all_words.extend(new_words)

    print(f"  Transcription complete: {len(all_words)} words")
    return all_words
