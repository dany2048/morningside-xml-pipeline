# Morningside XML Pipeline

Auto-generates rough cuts from raw talking-head YouTube footage. Takes a 30-60 min raw recording and produces an FCPXML file for Premiere Pro with only the "keeper" segments.

## How It Works

1. **Audio extraction** — ffmpeg pulls audio from the raw MP4
2. **Transcription** — OpenAI Whisper produces word-level timestamps
3. **LLM analysis** — GPT-5.4 reads the full numbered transcript and decides which lines to KEEP vs CUT
4. **FCPXML output** — Kept segments become clips in an FCP7 XML file, importable into Premiere Pro

The LLM prompt includes structural rules learned from RLHF reviews (human-corrected cut decisions on past footage).

## Pipeline Versions

| File | Model | Strategy | Notes |
|---|---|---|---|
| `processor.py` | GPT-4o | 3-pass (aggressive cut, dedup, polish) | Original |
| `processor_v2.py` | GPT-5.4 | Single-pass with structural rules | Current, ~$0.23/run |

## Test Output Versions (C5296)

All XML versions in `test-outputs/` for comparison:

| File | Prompt version | Duration kept |
|---|---|---|
| `Clean Cut.xml` | GPT-4o 3-pass, no RLHF | 765s (40%) |
| `Clean Cut v2.xml` | GPT-5.4 3-pass, no RLHF | baseline |
| `Clean Cut fewshot-rlhf.xml` | GPT-5.4 single-pass + transcript-specific few-shot | 963s (50%) |
| `Clean Cut structural-rules.xml` | GPT-5.4 single-pass + generalizable structural rules | 872s (45%) |

Published video: ~14:38 (~878s, 46%). The structural-rules version (872s/45%) is the closest.

## RLHF Review Process

1. Run pipeline on test footage -> produces KEEP/CUT decisions
2. Export to a review file with `-->` annotation fields
3. Human reviews every line, marks WRONG decisions with corrections
4. Error patterns get folded into the prompt as structural rules

Review files in `rlhf-reviews/`.

## Setup

```bash
cd scripts/morningside-xml-pipeline
pip install openai python-dotenv
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

Requires ffmpeg for audio extraction: `brew install ffmpeg`

## Usage

```bash
# Full pipeline (raw MP4 -> FCPXML)
python3 main_v2.py /path/to/raw.mp4

# Test on cached C5296 data (auto-increments version)
python3 test_v2.py
python3 test_v2.py --tag my-test

# RLHF review from external transcript
python3 rlhf_from_transcript.py /path/to/transcript.srt
```

## Importing into Premiere Pro

1. File > Import > select the `.xml` file
2. Premiere will prompt you to locate the source MP4
3. The timeline has only the clean takes, in order

## Tuning

Edit `config.py` for thresholds (repeat similarity, pause lengths, filler words, segment padding).
