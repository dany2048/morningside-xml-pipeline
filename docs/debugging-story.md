# The Two-Bug Debugging Story

A write-up of the April 13-14 2026 session where the pipeline went from "cuts don't match at all" to "near-damn-perfect in Premiere on the first import." Written in the first person (Danyal's) with technical assist from Claude Code (Opus 4.6).

## 1. The starting symptom

I had a pipeline that took a raw MP4, ran it through OpenAI's Whisper API for word-level transcription, fed the transcript to GPT-5.4 to pick which lines to keep, and emitted a FCP7 XML for Premiere to import. The review txt file it generated looked correct. When I imported the XML into Premiere, the cuts landed on completely wrong frames.

I had rebuilt the XML generator four times before this session. Same result each time.

The test file was `20260203_C4109 (shorter clip) .MP4` — a 5.4 minute clip shot on a Sony FX3 at 4K, 23.976fps. I had a hardcoded `KEEP_RANGES` list of word ID pairs that represented the correct cuts:

```python
KEEP_RANGES = [
    (36, 61), (62, 85), (86, 136), (158, 199),
    (321, 364), (396, 414), (498, 526), (527, 530),
]
```

My ask to Claude Code was simple: rebuild the XML generator from scratch, test it against this keep list, tell me what's wrong.

## 2. The first rebuild (wrong fix, right outcome)

Claude wrote `test_c4109_v2.py` — a fresh standalone XML generator that bypassed `xml_gen.py` entirely. Single sequence, clean integer frame math at `24000/1001`, explicit `<in>`/`<out>`/`<start>`/`<end>`, no nest indirection. The output parsed cleanly. The segment frame numbers matched what the keep ranges said they should be. `c4109_v2_cut.xml` got generated.

I imported it into Premiere. Still broken. Cuts still didn't match.

But in hindsight this was useful. It isolated the variable. If the clean-room generator produced the same wrong result as the old one, the bug wasn't in my XML-writing code. It had to be either (a) in how Premiere interprets the XML, or (b) in the input data being fed to the generator.

## 3. The ffprobe moment

Claude ran `ffprobe -v error -show_streams -show_format` on the source MP4 before writing another line of code. Buried in the data stream output:

```
--- data None
  codec_tag_string: rtmd
  tag[timecode]: 03:35:01:00
```

**That was the whole bug for Premiere side.**

Sony XAVC cameras record with an embedded start timecode in the MP4. The C4109 file starts at 03:35:01:00 from the camera's perspective, not at 00:00:00:00. This is standard for any broadcast/cinema camera — they use timecode for multicam sync and shot log continuity.

The Apple FCP7 XML Interchange Format spec is unambiguous: `<clipitem>` `<in>` and `<out>` values are frame offsets **from the source file's embedded start timecode**, not from frame zero of the media. In the spec's own words:

> "To convert `in` and `out` to timecode values, use the starting timecode in the media, if there is one. For media without timecodes, the default starting timecode is `00:00:00:00`."

My XML was writing `<in>1079</in>` — which is correct for Whisper's "45.0 seconds into the media" coordinate. But the `<file>` element had no `<timecode>` child declaring the source's actual start TC. So Premiere was doing this:

- XML side: "this file starts at 00:00:00:00 because no timecode was declared"
- Media side: "this file's embedded TC says 03:35:01:00"

The two coordinate systems were about 309,624 frames out of phase. Every cut landed in the wrong place.

**The fix**: add a `<timecode>` child to the `<file>` element with the actual source start TC:

```xml
<file id="file-1">
  <name>C4109.MP4</name>
  <pathurl>file://localhost/path/to/C4109.MP4</pathurl>
  <rate>
    <timebase>24</timebase>
    <ntsc>TRUE</ntsc>
  </rate>
  <duration>7740</duration>
  <timecode>
    <rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
    <string>03:35:01:00</string>
    <frame>309624</frame>
    <displayformat>NDF</displayformat>
    <source>source</source>
  </timecode>
  <media>...</media>
</file>
```

The `<frame>` value uses integer-timebase math: `(3*3600 + 35*60 + 1) * 24 + 0 = 309624`. Not multiplied by 23.976 — use the integer 24, then `<ntsc>TRUE</ntsc>` separately tells Premiere each frame is 1001/24000 seconds of real time.

This bug is **invisible on phone-recorded test files** because phones don't embed TC. It only shows up on real camera footage. If you never test with actual production raws, you'd never find it.

Claude patched the generator with the fix, produced `c4109_v3_cut.xml`, and I imported it into Premiere.

## 4. Still broken, but now a different broken

The cuts *still* didn't match the review. But this time I noticed something independently: when I scrubbed through the raw file with the review txt open, the Whisper word-level timestamps in the review didn't line up with the actual audio either. Words were supposedly spoken at times when there was no speech. The transcript layer was lying to me.

Claude dumped the raw cached `c4109_words.json` as a human-readable `c4109_whisper_raw.txt` and we looked at the worst offenders:

| Word | Whisper says | "Duration" |
|---|---|---|
| `"right"` | 31.56s → 38.90s | **7.34s** |
| `"to"` | 50.96s → 65.92s | **14.96s** |
| `"for"` | 74.86s → 81.28s | **6.42s** |
| `"been"` | 146.34s → 160.74s | **14.40s** |
| `"video"` | 174.46s → 183.98s | **9.52s** |

No human speaks a 15-second word. What Whisper was doing: attributing all the silence that *followed* each word to that word's `end` timestamp. If the speaker said `"to"` and then paused for 15 seconds, Whisper reported the word `"to"` as a 15-second-long event.

20 words out of 530 had this failure mode in C4109.

## 5. "Is that really raw Whisper data?"

I asked Claude to verify. Was the cached JSON actually what the API returned, or had something in our pipeline mangled it? Good skeptical question — cheap to answer, expensive if we guessed wrong.

Claude ran a fresh OpenAI Whisper API call on the exact same audio file and compared the output against the cache. Two findings:

1. **The first 21 words matched byte-for-byte** — same words, same start, same end, zero drift. The cache was genuinely from the API. Our pipeline wasn't corrupting anything in between.

2. **The fresh call returned 399 words instead of 530.** Same audio, same model, same parameters, same `verbose_json` + `word` granularity. Different run → different transcript length. Whisper split the speaker's repeated takes differently on each call. **OpenAI's Whisper API is non-deterministic across runs.**

3. **The silence-bleed bug was present in the fresh call too** — 28 words with duration > 1.5 seconds this time, versus 20 in the cache. Reproducible on the live API against their current model.

So: the cache was real Whisper output, and Whisper itself was the broken layer. No amount of XML-side fixing could compensate because the source-of-truth timing data was already wrong. Worse, even if I fixed the pipeline to work around today's bad timestamps, the next run would produce a different set of bad timestamps.

## 6. Why Whisper's word timestamps are architecturally bad

Whisper is an encoder-decoder transformer. The encoder takes audio and produces feature representations. The decoder generates text tokens, using cross-attention to "look at" the encoder's audio features while it writes.

Whisper is not natively a word-level timestamp model. It's a transcription model. When you ask it for word-level timestamps, it computes them as a **post-processing heuristic**: for each generated word, look at the decoder's cross-attention weights and infer which audio timesteps the model was attending to when it emitted that word. Pick the peak. Call that the word's timestamp.

This is not acoustic measurement. It's inference from attention bookkeeping. Failure modes:

- **Silence bleed**: the decoder's attention lingers past the actual end of a spoken word into the following silence, so the word's `end` timestamp drifts forward.
- **Non-determinism**: Whisper uses temperature-based sampling with beam search and isn't deterministic in practice even at temperature 0.
- **Long-audio drift**: errors compound across 30-second chunks.

OpenAI's own documentation describes word timestamps from `whisper-1` as "approximate." They mean it.

## 7. Premiere's built-in transcription uses the right technique

I wondered: Premiere Pro's built-in Speech-to-Text feature produces accurate word-level timestamps that line up with the audio in the Text panel. What's it doing that Whisper API isn't?

Claude researched Adobe's documentation, community forums, and scripting APIs. Findings:

- Premiere's transcription uses an undisclosed Adobe Sensei model
- Behavior is consistent with **forced alignment** post-processing (word boundaries snap to actual phoneme on/offset, no silence bleed)
- There is **no API, no CLI, no ExtendScript hook, no UXP API, no AppleScript hook** to trigger transcription programmatically. As of early 2026, Adobe engineers confirmed on the community forum that the Text panel is GUI-only.
- Word-level data lives in Premiere's proprietary `.prtranscript` format; exports are either `.csv` (line-level) or `.srt` (segment-level). Not directly useful for my pipeline.

So using Premiere as the transcription backend was out. But the key phrase was **forced alignment**, and that's a well-understood technique with open-source implementations.

## 8. Forced alignment and WhisperX

Forced alignment is the classical speech processing technique for finding exact word boundaries. Given an audio clip and a known transcript, you find the exact acoustic time when each word is spoken. The method:

1. Run an acoustic model that produces per-frame phoneme probabilities (one probability distribution per ~20ms of audio)
2. Use dynamic programming (Viterbi decoding) to find the most likely alignment of the transcript's phonemes against the acoustic probabilities
3. Read off the word boundaries from the alignment

The key insight: word boundaries come from **acoustic evidence** (actual phoneme energy and spectral patterns in the audio waveform) rather than from a text-generating model's internal attention bookkeeping.

**WhisperX** (m-bain/whisperX on GitHub) is built for this exact use case. It runs Whisper for transcription and then runs a **wav2vec2** CTC model over the audio to force-align the transcript against the waveform.

wav2vec2 is Facebook's self-supervised audio representation model. It's pre-trained on raw audio using masked prediction (the audio equivalent of BERT's masked language modeling), then fine-tuned with CTC loss on labeled speech to become a phoneme classifier. Unlike Whisper, wav2vec2 produces one probability distribution per ~20ms of audio — phoneme by phoneme, densely in time. This is exactly what forced alignment needs.

The WhisperX flow:

```
audio.wav → Whisper → transcript text
            wav2vec2 → per-frame phoneme probabilities
            ↓
         dynamic programming align(text, probs)
            ↓
         word boundaries snapped to actual phoneme on/offset
```

Step 1 is "what was said." Step 2 is "when exactly it was said." They're complementary. Whisper is optimized for transcription accuracy; wav2vec2 is optimized for acoustic time alignment.

This is, based on my research, the same architectural approach Adobe Sensei uses inside Premiere's Text panel.

## 9. Installing WhisperX

Claude created a Python 3.9 venv (`.whisperx_venv`), ran `pip install whisperx`, and pulled down whisperx 3.7.5 + torch 2.8.0 + pyannote-audio + about 130 other packages. First run failed with this error:

```
_pickle.UnpicklingError: Weights only load failed. 
WeightsUnpickler error: Unsupported global: 
GLOBAL omegaconf.listconfig.ListConfig was not an allowed global by default.
```

Known PyTorch 2.6+ compatibility issue. PyTorch 2.6 changed the default of `torch.load`'s `weights_only` parameter from `False` to `True` to prevent arbitrary code execution during model loading. Pyannote's VAD model checkpoints contain `omegaconf.listconfig.ListConfig` objects that aren't in PyTorch's allowlist, so loading fails.

Fix: monkey-patch `torch.load` to force `weights_only=False` before importing whisperx. Pyannote is a trusted HuggingFace model so it's safe:

```python
import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

import whisperx
```

The `kw["weights_only"] = False` matters — `setdefault` doesn't work here because pytorch-lightning's loader passes `weights_only=True` explicitly as a keyword.

## 10. First WhisperX run results

Extracted audio as 16kHz mono WAV, loaded the `base` model on CPU with `int8` compute type, transcribed (Whisper pass, 27 seconds), force-aligned (wav2vec2 pass, 33 seconds). Total 65 seconds for a 5-minute clip.

| Metric | Whisper API | WhisperX | Change |
|---|---|---|---|
| Total words | 530 | **602** | +14% (finer granularity) |
| Median word duration | 220ms | **121ms** | **-45%** |
| Mean word duration | 438ms | 262ms | -40% |
| Max word duration | **14.96s** | 10.59s | -29% |
| Words > 1.5s duration | 20 | **11** | -45% |

The median dropping from 220ms to 121ms is the key number. That's what you'd expect when you stop baking silence into word ends.

Of the 11 remaining "long" words in the WhisperX output, 4 had a gap to the next word of 40ms or less — meaning the long "duration" is actual slow-spoken speech, not silence bleed. The other 7 all end in punctuation tokens (`'Palantir.'`, `'video.'`, `'Help.'`, `'is.'`) — these are end-of-sentence words where wav2vec2's alignment is slightly loose because there's no immediate next phoneme to anchor against. Residual pattern, not the old bug, and much easier to post-process.

## 11. End-to-end test and the breakthrough

Claude wrote `test_c4109_e2e.py` combining everything:
- Reads the WhisperX words cache
- Probes the MP4 via ffprobe for source TC, duration, resolution, audio channels
- Builds word-level numbered lines via `processor_v2._build_numbered_lines(word_level=True)`
- Calls `processor_v2.process_lines()` → GPT-5.4 single pass → keep list
- Generates FCP7 XML via a clean inline generator with the source TC fix baked in
- Writes a matching RLHF review txt

Results on C4109 (5.4 min):
- 602 forced-aligned words
- 223/602 words kept
- 22 clipitems on the timeline
- 52.7 seconds kept from 322.8 seconds raw (16.3%)
- Cost: $0.0875 (GPT-5.4 only — WhisperX is free local)
- Runtime: 58 seconds (GPT-5.4 pass; WhisperX was already cached)

I imported `c4109_v4_cut.xml` into Premiere. Every cut landed on the right frame. Audio and video synced across every segment. No silence bleed, no coordinate-system offset.

"Near-damn-perfect." The best version I've gotten by a long margin.

## 12. What worked vs what didn't

### What worked
- **Running ffprobe on the source MP4 first.** This should have been step zero of every debugging session. The XAVC timecode was the single most important fact and it was sitting in the file metadata the whole time.
- **Exporting raw data as readable txt files.** Dumping `c4109_whisper_raw.txt` made the silence-bleed bug visually obvious in a way that staring at JSON would not have been.
- **Cross-checking by re-running the API.** "Is this really from Whisper?" was the right skeptical question. Cheap to verify, revealed non-determinism as a bonus.
- **Running research agents in parallel** on FCP7 XML behavior and transcription alternatives. Saved round trips, kept main context clean.
- **Rebuilding the XML generator from scratch** rather than patching the accumulated cruft in `xml_gen.py`. Let me reason about the structure from first principles.
- **Swapping the transcription layer entirely** rather than compensating downstream. Every prior attempt tried to paper over Whisper's flaws with end-capping heuristics. The real fix was replacing the layer.

### What didn't work
- **Patching `xml_gen.py` without understanding the source TC issue.** All prior rebuild attempts were changing things downstream of the real bug.
- **Trusting Whisper's word timestamps without verification.** The existing `MAX_WORD_DUR = 1.0` cap was a hint that someone knew something was wrong with word ends, but it never went the extra step to ask "are the word *starts* also wrong?" (They often were.)
- **Nested sequences in the XML.** An earlier session added a `NEST + Clean Cut` structure to support external mic audio. Turned out unrelated to the real bugs and added complexity. Simpler single-sequence XML works fine.
- **Treating Whisper non-determinism as "sampling noise."** It's actually a deal-breaker for a pipeline that needs reproducibility.

### The meta-lesson
When a pipeline has multiple layers and the output is wrong, the instinct is often to rewrite the layer you understand best — for me, XML generation, because it's pure code with no external dependencies. But the real bug is frequently in the layer you're *trusting* — the source data (Whisper's timestamps) or the input file itself (embedded TC). **Probe everything before rewriting anything.**

## 13. Mental model for future sessions

When the pipeline produces wrong output, check layers in this order:

1. **Source file first.** `ffprobe -show_streams -show_format` on every new raw file. Check `r_frame_rate`, `duration`, `tag[timecode]`, `channels`, `sample_rate`. If anything is unexpected (non-zero TC, weird frame rate, mono audio, drop-frame TC), note it before writing any code.
2. **Transcription output second.** Dump raw words to a human-readable txt and look at them. Are word durations reasonable (0.05s–1.0s for normal speech)? Do word starts match the audio when you scrub to them?
3. **Segment building third.** Log the keep-range → time-segment conversion with exact frame numbers.
4. **XML generation last.** Write out, parse back, validate structure programmatically, then import into Premiere.

Prior sessions inverted this: started at layer 4 and never audited layers 1-2. Every rebuild was treating symptoms.

## Credits

Diagnostic session with Claude Code (Opus 4.6, 1M context) on April 13-14 2026. Claude ran ffprobe, dumped raw Whisper output, cross-checked API calls, researched FCP7 xmeml timecode semantics and Premiere's built-in transcription internals, installed WhisperX, wrote the end-to-end test. I drove the session, asked the skeptical questions, and verified each fix in Premiere. Total session time: about 3 hours across two days.
