# Morningside XML Pipeline

Automated rough-cut generator for long-form talking-head footage. You point it at a raw MP4, it figures out which takes are keepers and which are junk, and hands you back an FCP7 XML that drops straight into Premiere Pro with only the clean segments.

A 5-minute raw clip goes through the whole pipeline in about 60 seconds and costs about $0.09. A 30-minute clip is closer to 5–8 minutes and about $0.25. Nothing on my Mac needs a GPU.

## What it actually does

```
raw.mp4
  → ffmpeg extracts audio
  → WhisperX transcribes + forced-aligns every word to the waveform
  → GPT-5.4 reads the transcript and picks which words belong in the final cut
  → a FCP7 xmeml generator writes an XML that Premiere imports as a fresh sequence
  → you drag the source MP4 into Premiere, relink, and the timeline is already cut
```

The pipeline does not try to replace a human editor. It gives you a rough cut with the obvious junk (stutters, restarts, off-camera talk, repeated takes) removed, so the human can start from minute 5 instead of minute 0.

## How I got here (the short version)

I built the first version of this in early April 2026 using OpenAI's Whisper API for transcription and GPT-4o (then GPT-5.4) for the keep/cut logic. It worked end-to-end and the review txt I was generating looked correct. But every time I imported the XML into Premiere, the cuts landed on the wrong frames. Completely wrong. Like, the cuts had no relationship to what the review said should be kept.

I rebuilt the XML generator four times. Same result each time.

On April 13-14 I ran a proper debugging session with Claude Code as my pair. The report in `docs/debugging-story.md` walks through the full diagnosis, but the TL;DR is that two independent bugs were compounding:

**Bug 1: Sony XAVC timecode.** My test clips were filmed on a Sony FX3, and Sony cameras embed a start timecode in the MP4 (mine was `03:35:01:00`). In FCP7 XML, the `<clipitem>` `<in>` and `<out>` values are frame offsets **from the source file's embedded start timecode**, not from frame zero of the media. If the XML doesn't declare the source TC explicitly via a `<timecode>` child on `<file>`, Premiere falls back to `00:00:00:00` on one side and reads the embedded MP4 TC on the other. The two coordinate systems end up about 309,000 frames out of phase. Every cut lands in the wrong place.

This bug is invisible if you test with a phone recording (phones don't embed TC). It only shows up on real camera footage. Fix: add a `<timecode>` child to the `<file>` element declaring the source's actual start TC.

**Bug 2: Whisper API word timestamps are broken.** OpenAI's `whisper-1` generates word-level timestamps from cross-attention weights, which is a heuristic, not forced alignment. It bakes the silence that follows a word into that word's `end` timestamp. On my C4109 test file, the word `"to"` registered as 14.96 seconds long because the speaker paused after saying it. The model also isn't deterministic: same audio, same model, same parameters, returns different word counts across runs (530 vs 399 on C4109).

No amount of XML fixing could compensate for this because the source-of-truth timing data was already wrong. Fix: replace Whisper API with **WhisperX**, which runs Whisper for transcription and then runs a wav2vec2 CTC model over the audio waveform to force-align the words against real phoneme boundaries. Same architectural approach Adobe's Speech-to-Text uses internally inside Premiere. WhisperX is free, local, and runs on CPU on Apple Silicon in about 65 seconds for a 5-minute clip.

After fixing both bugs, the pipeline produced a cut that lined up frame-accurately in Premiere on the first try.

## How the technology stack fits together

Four moving parts:

**1. WhisperX** for transcription with accurate word timestamps.
Whisper (the OpenAI model) is good at producing the text of what was said. It's bad at saying exactly when each word was said, because its timestamps come from attention-weight heuristics rather than from the audio itself. WhisperX glues Whisper to a **wav2vec2** model (Facebook's self-supervised acoustic representation network), then uses dynamic programming to align the transcript against the wav2vec2 output. The wav2vec2 model produces a phoneme probability distribution every ~20ms of audio, so the alignment snaps word boundaries to real acoustic onset/offset. Result: median word duration on my test file dropped from 220ms (Whisper API) to 121ms (WhisperX), and the "15-second `to`" problem vanished entirely.

**2. GPT-5.4** for the keep/cut reasoning.
Once you have a numbered transcript with accurate timestamps, the semantic question is "which of these lines belong in the final video?" I feed the whole transcript into GPT-5.4 as a single pass with a prompt that encodes three structural rules learned from RLHF review sessions on prior footage:
- **Take grouping**: divide the transcript into sections (intro, main arg, CTA), identify re-attempts within each section, keep only the last complete attempt.
- **Orphan check**: after picking the keeps, re-read them as a sequence and remove any that start mid-sentence or reference something that was cut.
- **Visual references**: bias toward keeping lines that say "this graph" or "as you can see" because those carry on-screen context the transcript alone doesn't capture.

Reasoning effort is set to `medium`. Higher reasoning caused cascading over-cuts in earlier experiments. GPT-5.4 returns a JSON array of line numbers to keep.

**3. A clean FCP7 xmeml v5 generator** writes the XML.
FCP7 xmeml is Apple's legacy Final Cut Pro 7 interchange format. Premiere still supports it and it's my preferred output because it maps cleanly onto integer frame math at 23.976fps (timebase=24, ntsc=TRUE). The generator writes a single flat sequence, one `<clipitem>` per kept segment, with `<in>`/`<out>` as source frames and `<start>`/`<end>` as timeline frames. The critical bit is the `<timecode>` child on `<file id="file-1">` declaring the source's start TC so Premiere's coordinate system lines up with mine.

**4. ffprobe** for media metadata.
Before doing anything, the pipeline shells out to ffprobe to pull the source MP4's frame rate, resolution, audio channel count, duration, and — most importantly — the embedded timecode string. If you skip this step you get Bug 1.

## Setup

```bash
git clone git@github.com:dany2048/morningside-xml-pipeline.git
cd morningside-xml-pipeline

# ffmpeg is required for audio extraction
brew install ffmpeg

# WhisperX needs its own venv because of torch + pyannote dependencies
python3 -m venv .whisperx_venv
source .whisperx_venv/bin/activate
pip install --upgrade pip
pip install whisperx openai python-dotenv
deactivate

# API key for GPT-5.4
cp .env.example .env
# add OPENAI_API_KEY=sk-proj-... to .env
```

First run downloads ~360 MB of wav2vec2 model weights and ~140 MB of pyannote VAD weights. Cached after that.

**Known gotcha**: PyTorch 2.6+ defaults `torch.load(weights_only=True)`, which breaks pyannote's VAD checkpoint loading with an `omegaconf.listconfig.ListConfig` pickle error. The WhisperX runner in this repo monkey-patches `torch.load` to force `weights_only=False` before the import. If you see that pickle error, check the top of `run_whisperx_c4109.py` for the workaround.

## Usage

```bash
# Step 1: run WhisperX on the raw MP4 (slow, but cached — only run once per clip)
source .whisperx_venv/bin/activate
python run_whisperx_c4109.py
deactivate

# This caches forced-aligned words to outputs/rlhf/c4109_words_whisperx.json
# Takes ~65 seconds for a 5-min clip on M-series CPU. Longer clips scale linearly.

# Step 2: run the end-to-end pipeline (WhisperX cache + GPT-5.4 + clean XML)
python3 test_c4109_e2e.py

# Outputs:
#   outputs/rlhf/c4109_v4_cut.xml     ← import this into Premiere
#   outputs/rlhf/c4109_v4_review.txt  ← word-by-word keep/cut review for debugging
```

The two scripts are currently hardcoded to C4109 for the development loop. Generalizing them to arbitrary input files is the next refactor. For now, swap the paths at the top of each script to point at the clip you want to process.

### Importing into Premiere

1. File → Import → select the `.xml` file
2. Premiere prompts you to locate the source MP4. Point it at the original.
3. A new sequence called `<filename> - Clean Cut v4` appears in your project panel.
4. Open it. The timeline has only the kept segments, in order, cut at frame-accurate word boundaries.

The XML references the source MP4 directly so Premiere will scrub/play it natively. No proxies, no rendering, no sync step. If you want to add external mic audio later, you can create a nested sequence manually in Premiere and drop the lav track in — the v3 "NEST + Clean Cut" dual-sequence structure was dropped after the source-TC fix made it unnecessary.

## Current state

**What works**: End-to-end on C4109 (5.4 min, Sony FX3, 4K, 23.976fps). Near-perfect cuts confirmed in Premiere.

**What needs work**:
- `main_v2.py` (the production CLI entry point) still uses the old Whisper API + the pre-fix `xml_gen.py`. The breakthrough lives in `test_c4109_e2e.py`. Migrating the fix back into the canonical entry point is the next cleanup.
- Scripts are hardcoded to C4109. Generalizing to `python run.py raw/<any>.mp4` is next.
- Only tested on one clip. C4340 (27 min) and C5296 (32 min) are the next scaling targets.
- WhisperX on Apple Silicon runs CPU-only because torch's MPS backend crashes pyannote. That's fine at current scale but means longer clips take longer. For a 27-min clip expect ~6-8 minutes of WhisperX time.

## Repo layout

```
morningside-xml-pipeline/
├── README.md                    ← this file
├── docs/
│   └── debugging-story.md       ← full report of the two-bug debugging session
├── config.py                    ← shared constants (fps table, thresholds)
├── audio.py                     ← ffmpeg/ffprobe wrappers
├── transcribe.py                ← Whisper API transcription (LEGACY, do not use)
├── run_whisperx_c4109.py        ← WhisperX runner (CURRENT)
├── processor.py                 ← GPT-4o original 3-pass (LEGACY)
├── processor_v2.py              ← GPT-5.4 single-pass with structural rules (CURRENT)
├── xml_gen.py                   ← FCP7 XML generator (PARTIAL FIX, needs source-TC rebuild)
├── test_c4109_e2e.py            ← end-to-end test with all fixes applied inline (CURRENT)
├── main_v2.py                   ← production CLI (NEEDS MIGRATION to the fixed path)
├── rlhf_capture.py              ← dumps KEEP/CUT decisions for human review
├── rlhf_from_transcript.py      ← parses Premiere-exported transcripts (Mode B)
├── drive.py / notion_handler.py ← Google Drive + Notion integration for Morningside ops
├── requirements.txt
└── .whisperx_venv/              ← gitignored, local Python env
```

## Credits

I built this with [Claude Code](https://claude.com/claude-code) as my pair, running Opus 4.6. The debugging session that cracked the two compounding bugs was a back-and-forth where Claude ran ffprobe on the source file, dumped raw Whisper output into readable txt files, cross-checked the cache against fresh API calls, researched FCP7 xmeml timecode behavior in parallel with researching forced-alignment alternatives to Whisper, and eventually installed + ran WhisperX locally. The full story is in `docs/debugging-story.md`.

The keep/cut logic builds on RLHF review work from earlier sessions — every wrong decision GPT-5.4 made on past clips got turned into a structural rule in the prompt. The three rules (take grouping, orphan check, visual reference bias) are the current best generalization.

The test clips come from Morningside AI, who I work with on content production. API credits come from their OpenAI account. The pipeline architecture is mine.
