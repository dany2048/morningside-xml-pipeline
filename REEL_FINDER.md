# Reel Finder

A new stage on top of the XML pipeline: point it at a **long interview / testimonial**
and it returns **ranked, bite-sized reel clips with timestamps** — built for marketing.

Where the keep/cut pipeline answers *"which takes are keepers?"*, the reel-finder answers
*"which 8-60s moments are postable reels?"* — specific results/numbers, before→after
transformation, emotional beats, objection flips, and quotable one-liners.

## Pipeline

```
raw interview .mp4
   │  run_whisperx_c4109.py --file <mp4> --model small      (word-level transcript)
   ▼
outputs/rlhf/<tag>_words_whisperx.json
   │  reel_finder.py <tag> --subject "Name" --context "..."  (GPT-5.4 full-coverage scan)
   ▼
outputs/reel-finder/<tag>_reels.json   (structured)
outputs/reel-finder/<tag>_reels.md     (ranked, human-readable + paste-ready ranges)
```

The whole transcript goes to GPT-5.4 in one pass (1M context) — nothing is truncated, and
the prompt forces an exhaustive section-by-section scan so good moments aren't missed.

## Usage

```bash
# 1) transcribe (word-level, frame-accurate timing reusable for cuts)
source .whisperx_venv/bin/activate
python run_whisperx_c4109.py --file "/path/to/interview.mp4" --model small
deactivate

# 2) find reels  (tag = the *_words_whisperx.json stem, or pass the json path)
python3 reel_finder.py <tag> --subject "David Ortiz" \
    --context "AAA Accelerator member testimonial" --min 8 --max 60 --max-clips 30
```

Each clip in the output has: rank, score (1-10), category
(result/transformation/emotional/objection/quotable/insight/endorsement), an on-screen
**hook** (≤8 words), in/out timecodes + seconds, the **verbatim quote**, and **why** it works.

## Turning a clip into a cut (optional)

The `*_reels.md` ends with a paste-ready `REELS = {...}` dict of `(start, end)` seconds —
the exact shape `reels_from_ranges.py` consumes. Drop it in to render each reel as its own
Clean Cut sequence XML for Premiere.

## AAA testimonials batch

`find_testimonial_reels.sh` normalizes the WhisperX tags to clean names
(`timm_freeman`, `david_ortiz`, `josh_boucher`) and runs the finder on all three.
