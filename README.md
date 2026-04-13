# Morningside XML Pipeline

Automated rough-cut generator for long-form talking-head footage. You point it at a raw MP4, it figures out which takes are keepers and which are junk, and hands you back an FCP7 XML that drops straight into Premiere Pro with only the clean segments.

A 5-minute raw clip goes through the whole pipeline in about 60 seconds and costs about $0.09 (via GPT-5.4). A 30-minute clip takes about 4 minutes of pipeline time plus about 5 minutes of your time for the Claude Code keep/cut step, at zero marginal cost. Nothing on my Mac needs a GPU.

## What it actually does

```
raw.mp4
  → ffmpeg extracts audio
  → WhisperX transcribes + forced-aligns every word to the waveform
  → an LLM reads the transcript and picks which words belong in the final cut
      (Claude Opus 4.6 for clips > 10 min, GPT-5.4 for shorter clips — see below)
  → a FCP7 xmeml generator writes an XML that Premiere imports as a fresh sequence
  → you drag the source MP4 into Premiere, relink, and the timeline is already cut
```

The pipeline does not try to replace a human editor. It gives you a rough cut with the obvious junk (stutters, restarts, off-camera talk, repeated takes) removed, so the human can start from minute 5 instead of minute 0.

## Which LLM for the keep/cut step?

Short answer:

- **Clip is ≤10 minutes → use GPT-5.4.** Fully automated, ~$0.10 per run, good picks. Runs via the OpenAI API, no human in the loop.
- **Clip is >10 minutes → use Claude Opus 4.6.** GPT-5.4 falls apart on long content (see below). Claude scales cleanly to 30+ minute clips. Currently runs as a manual paste loop using the `--keeps-file` flag in `test_c4109_e2e.py`. Zero API cost.

Long answer (tested head-to-head on 2026-04-14):

On a 5.4-minute clip (C4109), GPT-5.4 and Claude are essentially interchangeable. Feeding the same WhisperX-aligned transcript to both models, they produced **identical outputs on 221 of 223 kept lines** — 99.1% overlap, same 22 segments, same 52.7s kept duration. The only difference was that Claude kept two extra semantically load-bearing words (`"contenders"` and `"Palantir."`) that GPT-5.4 cut because their WhisperX durations exceeded 1.5s and the reasoning model used timestamp suspicion as a heuristic cut signal. Either model works.

On a 27.6-minute clip (C4340), GPT-5.4 **catastrophically collapses**. It kept 88 of 3995 lines (2.2%), and every single one was an isolated trailing word scattered across the whole transcript — words like `"taps."`, `"creations,"`, `"blueprint,"`. Not a "more aggressive cut." Literal garbage output. Claude on the same transcript picked 3254 of 3995 lines (81%) across 78 coherent take-group sections, producing a 13.3-minute final cut with a 48.2% keep ratio that lined up near-perfectly in Premiere on the first import.

The GPT-5.4 failure mode is a reasoning-token collapse. On C4340 it consumed 8609 of its 8924 output tokens on internal reasoning and emitted only ~315 tokens of actual JSON. Trying to recursively apply the "keep only the last attempt of each section" structural rule across 27 minutes compounded into absurdity, and the tiny remaining output budget was only enough to emit ~87 line IDs — so the model degraded to picking just the trailing word of each section as a shortcut. Same architectural failure mode as the earlier 3-pass experiments this project tried and abandoned.

Claude Opus 4.6 doesn't have this failure mode because it doesn't run reasoning-token cascades on this task. Plain "read the whole transcript in 1M-token context, apply the rules, emit the keep list at the end" pattern. Scales to any clip length without degrading.

**Practical rule:** under 10 minutes, GPT-5.4 is faster (fully automated, no human step). Over 10 minutes, Claude is the only one that works. Use whichever fits.

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

**2. Claude Opus 4.6 (long clips) or GPT-5.4 (short clips)** for the keep/cut reasoning.
Once you have a numbered transcript with accurate timestamps, the semantic question is "which of these lines belong in the final video?" I feed the whole transcript into an LLM as a single pass with a prompt that encodes three structural rules learned from RLHF review sessions on prior footage:
- **Take grouping**: divide the transcript into sections (intro, main arg, CTA), identify re-attempts within each section, keep only the last complete attempt.
- **Orphan check**: after picking the keeps, re-read them as a sequence and remove any that start mid-sentence or reference something that was cut.
- **Visual references**: bias toward keeping lines that say "this graph" or "as you can see" because those carry on-screen context the transcript alone doesn't capture.

The LLM returns a JSON array of line numbers to keep, which the pipeline feeds straight into the XML generator.

GPT-5.4 (via OpenAI API) is fully automated and handles short clips cleanly, but collapses on anything longer than ~10 minutes due to reasoning-token cascades. Claude Opus 4.6 (via the Claude Code session directly, using the `--keeps-file` flag) doesn't have this failure mode and scales to any length. See the "Which LLM for the keep/cut step?" section above for the full test results.

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

### Step 1: transcription (always the same, runs once per clip)

```bash
source .whisperx_venv/bin/activate
python run_whisperx_c4109.py --file reference/raw/C4109.MP4
deactivate
```

Caches forced-aligned words to `outputs/rlhf/<tag>_words_whisperx.json`. Takes about 65 seconds for a 5-minute clip on Apple Silicon CPU. Scales sublinearly — a 27-minute clip took 174 seconds in my C4340 test. First run also downloads ~500 MB of model weights (wav2vec2 + pyannote VAD).

### Step 2A: keep/cut picks — short clips, automated (GPT-5.4)

For clips ≤10 minutes:

```bash
python3 test_c4109_e2e.py --file reference/raw/C4109.MP4
```

This calls GPT-5.4 via the OpenAI API, feeds it the WhisperX transcript with the structural-rules prompt, gets back a JSON array of line IDs, and generates the FCP7 XML. Costs about $0.10 per run, takes about 60 seconds. Fully automated, no human step.

Outputs:
- `outputs/rlhf/<tag>_v4_cut.xml` — import this into Premiere
- `outputs/rlhf/<tag>_v4_review.txt` — word-by-word keep/cut review for debugging

### Step 2B: keep/cut picks — long clips, Claude Code loop

For clips >10 minutes, GPT-5.4 will collapse into garbage (see "Which LLM for the keep/cut step?" above). Use Claude Opus 4.6 via the Claude Code session directly. The workflow is:

1. Generate the numbered transcript from the WhisperX cache:
   ```bash
   python3 -c "
   import json, sys
   sys.path.insert(0, 'scripts/morningside-xml-pipeline')
   from processor_v2 import _build_numbered_lines, _format_for_llm
   words = json.load(open('outputs/rlhf/c4340_words_whisperx.json'))
   print(_format_for_llm(_build_numbered_lines(words, word_level=True)))
   " > /tmp/c4340_transcript_for_claude.txt
   ```

2. Open a Claude Code session and paste the transcript along with the prompt from `processor_v2.py` (the `SINGLE_PASS_PROMPT` constant, starting with "You are an expert video editor..."). Ask Claude to return a flat JSON array of line numbers to keep. Save the response as `outputs/rlhf/<tag>_keeps_claude.json`.

3. Run the e2e script with the `--keeps-file` flag to skip the LLM call and use your keeps directly:
   ```bash
   python3 test_c4109_e2e.py \
     --file reference/raw/C4340.MP4 \
     --keeps-file outputs/rlhf/c4340_keeps_claude.json \
     --tag v5-claude
   ```

Outputs land at `outputs/rlhf/<tag>_v5-claude_cut.xml` and `<tag>_v5-claude_review.txt`.

Zero API cost. Takes about 5 minutes of your time per clip for the paste-and-review step. Scales to any clip length Claude can read in context, which in practice is any clip you're likely to edit.

### Importing into Premiere

### Importing into Premiere

1. File → Import → select the `.xml` file
2. Premiere prompts you to locate the source MP4. Point it at the original.
3. A new sequence called `<filename> - Clean Cut <tag>` appears in your project panel.
4. Open it. The timeline has only the kept segments, in order, cut at frame-accurate word boundaries.

The XML references the source MP4 directly so Premiere will scrub/play it natively. No proxies, no rendering, no sync step. If you want to add external mic audio later, you can create a nested sequence manually in Premiere and drop the lav track in — the v3 "NEST + Clean Cut" dual-sequence structure was dropped after the source-TC fix made it unnecessary.

## Current state

**What works**:
- End-to-end on C4109 (5.4 min, Sony FX3, 4K, 23.976fps). Near-perfect cuts confirmed in Premiere with both GPT-5.4 and Claude Opus 4.6 pickers (99.1% overlap).
- End-to-end on C4340 (27.6 min, Sony FX3, 4K, **29.97fps**, source TC `05:26:43:24`). Near-perfect cuts confirmed in Premiere with Claude Opus 4.6 picker. GPT-5.4 failed on this clip (reasoning collapse, see above).
- Variable frame rate handling: one script handles 23.976 / 24 / 25 / 29.97 / 30 / 50 / 59.94 / 60 via ffprobe.
- Variable source timecode handling: works for any Sony XAVC clip regardless of start TC value.
- Manual Claude Code loop via `--keeps-file` flag for clips that exceed GPT-5.4's effective range.

**What needs work**:
- `main_v2.py` (the production CLI entry point) still uses the old Whisper API + the pre-fix `xml_gen.py`. The breakthrough lives in `test_c4109_e2e.py`. Migrating the fix back into the canonical entry point is the next cleanup.
- No automated Anthropic API integration yet. Adding a `processor_claude.py` that wraps the Anthropic SDK would remove the manual paste step from the long-clip workflow. Would need `ANTHROPIC_API_KEY` in `.env` and cost about $0.50 per 30-min clip at Opus pricing.
- WhisperX on Apple Silicon runs CPU-only because torch's MPS backend crashes pyannote. That's fine at current scale but means longer clips take longer. A 27-min clip takes about 3 minutes of WhisperX time.

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
