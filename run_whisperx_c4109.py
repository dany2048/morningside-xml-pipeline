#!/usr/bin/env python3
"""Run WhisperX on C4109 to get forced-aligned word-level timestamps.

WhisperX = Whisper transcription + wav2vec2 forced alignment. Solves the
Whisper API's silence-bleed-into-word-end bug by snapping word boundaries
to actual phoneme on/offset from the audio waveform.

Outputs:
  outputs/rlhf/c4109_words_whisperx.json  — same shape as Whisper API cache
  outputs/rlhf/c4109_whisperx_raw.txt     — readable dump for eyeballing
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

# WhisperX/pyannote ships VAD model checkpoints that include omegaconf
# objects, which PyTorch 2.6+ refuses to deserialize under the new
# weights_only=True default. The pyannote VAD model is a trusted HF
# checkpoint, so we restore the legacy behavior before importing whisperx.
import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

import whisperx

WORKSPACE = Path("/Users/dany/Documents/Claude Workspaces/personal-workspace")
SRC_MP4 = WORKSPACE / "reference/Raw Files Tests/20260203_C4109 (shorter clip) .MP4"
OUT_JSON = WORKSPACE / "outputs/rlhf/c4109_words_whisperx.json"
OUT_TXT = WORKSPACE / "outputs/rlhf/c4109_whisperx_raw.txt"

# WhisperX needs CPU on Apple Silicon (no MPS support).
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # fastest on CPU; float32 is more accurate but slower
MODEL_SIZE = "base"    # base/small/medium/large-v2 — start small, escalate if needed


def extract_wav(mp4: Path) -> Path:
    """Extract 16kHz mono WAV — what wav2vec2 wants."""
    out = Path(tempfile.mkdtemp(prefix="wx_")) / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4),
         "-vn", "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(out)],
        capture_output=True, check=True,
    )
    return out


def main():
    print(f"Source: {SRC_MP4}")
    t0 = time.time()

    print("\n[1/4] Extracting 16kHz mono WAV...")
    wav = extract_wav(SRC_MP4)
    print(f"  → {wav}  ({wav.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n[2/4] Loading Whisper model ({MODEL_SIZE}, {DEVICE}, {COMPUTE_TYPE})...")
    t = time.time()
    model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE,
                                language="en")
    print(f"  loaded in {time.time()-t:.1f}s")

    print("\n[3/4] Transcribing (Whisper pass)...")
    t = time.time()
    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=8, language="en")
    print(f"  done in {time.time()-t:.1f}s — {len(result['segments'])} segments")

    print("\n[4/4] Forced alignment (wav2vec2 pass)...")
    t = time.time()
    align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio,
                             device=DEVICE, return_char_alignments=False)
    print(f"  done in {time.time()-t:.1f}s")

    # Flatten word_segments into the same shape as the Whisper API cache:
    # [{"word": "...", "start": float, "end": float}, ...]
    words = []
    for w in aligned.get("word_segments", []):
        # WhisperX may emit words without timestamps (failed alignment).
        # Skip those — they can't be used for cuts anyway.
        if "start" not in w or "end" not in w:
            continue
        words.append({
            "word": w["word"].strip(),
            "start": float(w["start"]),
            "end": float(w["end"]),
        })

    OUT_JSON.write_text(json.dumps(words, indent=2))
    print(f"\n✓ Wrote {len(words)} aligned words → {OUT_JSON}")

    # Readable dump
    lines = []
    bar = "=" * 80
    lines.append(bar)
    lines.append("WHISPERX FORCED-ALIGNED WORD TRANSCRIPT — C4109")
    lines.append(f"Model: {MODEL_SIZE} | Device: {DEVICE} | Compute: {COMPUTE_TYPE}")
    lines.append(f"Total words: {len(words)}")
    lines.append("Format: W#### [start - end] (dur) word")
    lines.append(bar)
    lines.append("")

    long_count = 0
    prev_end = 0.0
    for i, w in enumerate(words, 1):
        gap = w["start"] - prev_end
        if gap > 0.5 and i > 1:
            lines.append(f"  --- gap {gap:.2f}s ---")
        dur = w["end"] - w["start"]
        flag = ""
        if dur > 1.5:
            flag = "  <-- LONG"
            long_count += 1
        lines.append(f"W{i:04d} [{w['start']:7.3f} - {w['end']:7.3f}] ({dur:5.2f}s) {w['word']}{flag}")
        prev_end = w["end"]

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"✓ Wrote readable dump → {OUT_TXT}")
    print(f"\nWords with duration > 1.5s: {long_count}")
    print(f"  (Whisper API cache had 20)")
    print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
