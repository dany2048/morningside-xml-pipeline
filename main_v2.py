"""Morningside XML Pipeline v2 — GPT-5.4 powered.

Same pipeline as main.py but uses processor_v2.py (GPT-5.4) instead of processor.py (GPT-4o).
All other components (audio, transcribe, xml_gen) are shared.

Usage:
    # Full pipeline: MP4 → Whisper → GPT-5.4 → XML
    python main_v2.py --file /path/to/raw.mp4
    python main_v2.py --file /path/to/raw.mp4 --out /path/to/output.xml

    # Premiere transcript mode: skip Whisper, use Premiere's transcription
    python main_v2.py --transcript /path/to/transcript.txt --file /path/to/raw.mp4
    python main_v2.py --transcript /path/to/transcript.srt --file /path/to/raw.mp4

    # Notion + Drive mode (production — single page)
    python main_v2.py --notion-id <page_id>

    # Watch mode (production — polls Notion every 2 min)
    python main_v2.py --watch

    # Options
    python main_v2.py --file ... --local              # Use local Whisper instead of API
    python main_v2.py --file ... --whisper-model medium

Cost estimate (GPT-5.4 @ $2.50/1M in, $15.00/1M out):
    Single pass: ~25k in + ~2k out = ~$0.09
    Whisper API (if used): ~$0.36/video
    Grand total: ~$0.20-$0.45/video
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parent.parent
load_dotenv(_script_dir / ".env", override=True)
load_dotenv(_workspace_root / ".env")

import config
from audio import get_video_metadata, extract_audio, chunk_audio
from transcribe import transcribe_all
from processor_v2 import process, process_lines
from xml_gen import generate_fcpxml
from rlhf_from_transcript import detect_and_parse


def _run_core(mp4_path: str, output_path: str, use_local: bool = False, transcript_path: str | None = None) -> str:
    """Core pipeline: MP4 in, FCPXML out. Returns output path.

    If transcript_path is given, skips audio extraction + Whisper and uses
    the Premiere Pro transcript directly.
    """
    start_time = time.time()

    # Step 1: Video metadata
    print("\n[1] Analyzing video...")
    metadata = get_video_metadata(mp4_path)
    print(f"  Resolution: {metadata['width']}x{metadata['height']}")
    print(f"  FPS: {metadata['fps']}")
    print(f"  Duration: {metadata['duration_seconds']:.1f}s ({metadata['duration_seconds']/60:.1f} min)")
    print(f"  Codec: {metadata['codec_name']}")

    if transcript_path:
        # Premiere transcript mode — skip Whisper
        print(f"\n[2] Parsing transcript: {transcript_path}")
        lines = detect_and_parse(transcript_path)
        if not lines:
            raise RuntimeError("No lines parsed from transcript")
        total_duration = lines[-1]["end"]
        print(f"  {len(lines)} lines, {total_duration:.0f}s")

        print(f"\n[3] Processing transcript (GPT-5.4 single-pass)...")
        segments = process_lines(lines, total_duration)

    else:
        # Full pipeline — Whisper transcription
        with tempfile.TemporaryDirectory(prefix="morningside_v2_") as tmp_dir:
            print("\n[2] Extracting audio...")
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(mp4_path, audio_path)

            print("\n[3] Transcribing (Whisper)...")
            chunk_dir = os.path.join(tmp_dir, "chunks")
            chunks = chunk_audio(audio_path, chunk_dir)
            if len(chunks) > 1:
                print(f"  Split into {len(chunks)} chunks for Whisper")

            mode_label = "local Whisper" if use_local else "OpenAI Whisper API"
            print(f"  Using {mode_label}" + (f" (model: {config.WHISPER_MODEL})" if use_local else ""))
            words = transcribe_all(chunks, use_local=use_local)

            if not words:
                raise RuntimeError("No words detected in transcription")

            whisper_cost = metadata["duration_seconds"] / 60 * 0.006
            print(f"  Whisper cost: ${whisper_cost:.2f}")

            print(f"\n[4] Processing transcript (GPT-5.4 single-pass)...")
            segments = process(words, metadata["duration_seconds"])

    if not segments:
        raise RuntimeError("No segments survived processing")

    # Generate FCPXML
    print(f"\n[{'4' if transcript_path else '5'}] Generating FCPXML...")
    generate_fcpxml(segments, metadata, Path(mp4_path).name, output_path, source_path=mp4_path)

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s")
    if not transcript_path:
        print(f"  Estimated cost: ${whisper_cost:.2f} (Whisper) + GPT-5.4 (see above)")
    return output_path


def run_local(mp4_path: str, output_path: str | None = None, whisper_model: str | None = None,
               use_local: bool = False, transcript_path: str | None = None) -> str:
    """Local file mode."""
    mp4_path = os.path.abspath(mp4_path)
    if not os.path.exists(mp4_path):
        print(f"Error: File not found: {mp4_path}")
        sys.exit(1)

    if transcript_path:
        transcript_path = os.path.abspath(transcript_path)
        if not os.path.exists(transcript_path):
            print(f"Error: Transcript not found: {transcript_path}")
            sys.exit(1)

    source_filename = Path(mp4_path).stem
    file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)

    if whisper_model:
        config.WHISPER_MODEL = whisper_model

    if not output_path:
        tag = "transcript" if transcript_path else "v2"
        output_path = os.path.join(
            os.path.dirname(mp4_path),
            f"{source_filename} - Clean Cut {tag}.xml",
        )

    mode = "Premiere Transcript" if transcript_path else "Whisper"
    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline v2 (GPT-5.4 + {mode})")
    print(f"{'='*60}")
    print(f"Source: {mp4_path}")
    print(f"Size: {file_size_mb:.0f} MB")
    if transcript_path:
        print(f"Transcript: {transcript_path}")

    result = _run_core(mp4_path, output_path, use_local, transcript_path)

    print(f"\n{'='*60}")
    print(f"Output: {result}")
    print(f"{'='*60}")
    print(f"\nNext: Open Premiere Pro > File > Import > select the .xml")
    print(f"Premiere will ask you to relink media — point it to the original MP4.")
    return result


def run_notion(page_id: str, use_local: bool = False) -> str:
    """Notion + Drive mode."""
    from drive import parse_drive_file_id, download_file, upload_file
    from notion_handler import get_page, update_xml_property

    output_folder_id = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
    if not output_folder_id:
        print("Error: GOOGLE_DRIVE_OUTPUT_FOLDER_ID not set in .env")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline v2 (GPT-5.4) — Notion Mode")
    print(f"{'='*60}")

    print(f"\nReading Notion page {page_id}...")
    page = get_page(page_id)
    print(f"  Title: {page['title']}")

    if not page["raws_url"]:
        print("Error: No RAWs URL found on this Notion page")
        sys.exit(1)

    print(f"  RAWs URL: {page['raws_url']}")
    file_id = parse_drive_file_id(page["raws_url"])

    with tempfile.TemporaryDirectory(prefix="morningside_v2_") as tmp_dir:
        print("\nDownloading from Drive...")
        mp4_path = os.path.join(tmp_dir, "source.mp4")
        download_file(file_id, mp4_path)

        from drive import get_file_name
        source_name = get_file_name(file_id)
        source_stem = Path(source_name).stem

        output_path = os.path.join(tmp_dir, f"{source_stem} - Clean Cut v2.xml")
        _run_core(mp4_path, output_path, use_local)

        print("\nUploading FCPXML to Drive...")
        drive_url = upload_file(output_path, output_folder_id, f"{source_stem} - Clean Cut v2.xml")

    print("\nUpdating Notion...")
    update_xml_property(page_id, drive_url)

    print(f"\n{'='*60}")
    print(f"Complete! XML link written to Notion.")
    print(f"Drive URL: {drive_url}")
    print(f"{'='*60}")
    return drive_url


def run_watch(interval: int = 120, use_local: bool = False):
    """Watch mode — poll Notion DB every N seconds."""
    from notion_handler import get_ready_pages

    db_id = os.getenv("MORNINGSIDE_NOTION_DB_ID")
    if not db_id:
        print("Error: MORNINGSIDE_NOTION_DB_ID not set in .env")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline v2 (GPT-5.4) — Watch Mode")
    print(f"Polling every {interval}s for 'Started Editing' pages...")
    print(f"{'='*60}")

    while True:
        try:
            pages = get_ready_pages(db_id)
            if pages:
                print(f"\nFound {len(pages)} page(s) to process")
                for page in pages:
                    print(f"\n--- Processing: {page['title']} ---")
                    try:
                        run_notion(page["page_id"], use_local)
                    except Exception as e:
                        print(f"  Error processing {page['title']}: {e}")
                        continue
            else:
                now = time.strftime("%H:%M:%S")
                print(f"  [{now}] No pages ready. Checking again in {interval}s...")

        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
            break
        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Morningside XML Pipeline v2 (GPT-5.4)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="Path to local MP4 file")
    mode.add_argument("--notion-id", help="Notion page ID to process")
    mode.add_argument("--watch", action="store_true", help="Poll Notion DB for new pages")

    parser.add_argument("--transcript", help="Premiere Pro transcript file (.txt, .srt, .vtt) — skips Whisper")
    parser.add_argument("--out", help="Output FCPXML path (local mode only)")
    parser.add_argument("--local", action="store_true", help="Use local Whisper model instead of API")
    parser.add_argument("--interval", type=int, default=120, help="Watch mode poll interval in seconds (default: 120)")
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Local Whisper model size (default: base)",
    )
    args = parser.parse_args()

    if args.whisper_model:
        config.WHISPER_MODEL = args.whisper_model

    if args.file:
        run_local(args.file, args.out, args.whisper_model, use_local=args.local,
                  transcript_path=args.transcript)
    elif args.notion_id:
        run_notion(args.notion_id, use_local=args.local)
    elif args.watch:
        run_watch(interval=args.interval, use_local=args.local)


if __name__ == "__main__":
    main()
