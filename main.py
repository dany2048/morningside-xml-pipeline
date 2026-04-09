"""Morningside XML Pipeline — entry point and orchestrator.

Usage:
    # Local file mode (testing)
    python main.py --file /path/to/raw.mp4
    python main.py --file /path/to/raw.mp4 --out /path/to/output.fcpxml

    # Notion + Drive mode (production — single page)
    python main.py --notion-id <page_id>

    # Watch mode (production — polls Notion every 2 min)
    python main.py --watch

    # Options
    python main.py --file ... --local              # Use local Whisper instead of API
    python main.py --file ... --whisper-model medium
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Add script dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load project-local .env first (Morningside keys), then workspace .env as fallback
_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parent.parent
load_dotenv(_script_dir / ".env", override=True)
load_dotenv(_workspace_root / ".env")

import config
from audio import get_video_metadata, extract_audio, chunk_audio
from transcribe import transcribe_all
from processor import process
from xml_gen import generate_fcpxml


def _run_core(mp4_path: str, output_path: str, use_local: bool = False) -> str:
    """Core pipeline: MP4 in, FCPXML out. Returns output path."""
    start_time = time.time()

    with tempfile.TemporaryDirectory(prefix="morningside_") as tmp_dir:
        # Step 1: Video metadata
        print("\n[1/5] Analyzing video...")
        metadata = get_video_metadata(mp4_path)
        print(f"  Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  FPS: {metadata['fps']}")
        print(f"  Duration: {metadata['duration_seconds']:.1f}s ({metadata['duration_seconds']/60:.1f} min)")
        print(f"  Codec: {metadata['codec_name']}")

        # Step 2: Extract audio
        print("\n[2/5] Extracting audio...")
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(mp4_path, audio_path)

        # Step 3: Chunk + transcribe
        print("\n[3/5] Transcribing...")
        chunk_dir = os.path.join(tmp_dir, "chunks")
        chunks = chunk_audio(audio_path, chunk_dir)
        if len(chunks) > 1:
            print(f"  Split into {len(chunks)} chunks for Whisper")

        mode_label = "local Whisper" if use_local else "OpenAI Whisper API"
        print(f"  Using {mode_label}" + (f" (model: {config.WHISPER_MODEL})" if use_local else ""))
        words = transcribe_all(chunks, use_local=use_local)

        if not words:
            raise RuntimeError("No words detected in transcription")

        # Step 4: Process
        print("\n[4/5] Processing transcript...")
        segments = process(words, metadata["duration_seconds"])

        if not segments:
            raise RuntimeError("No segments survived processing")

        # Step 5: Generate FCPXML
        print("\n[5/5] Generating FCPXML...")
        generate_fcpxml(segments, metadata, Path(mp4_path).name, output_path, source_path=mp4_path)

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s")
    return output_path


def run_local(mp4_path: str, output_path: str | None = None, whisper_model: str | None = None, use_local: bool = False) -> str:
    """Local file mode — for testing or manual runs."""
    mp4_path = os.path.abspath(mp4_path)
    if not os.path.exists(mp4_path):
        print(f"Error: File not found: {mp4_path}")
        sys.exit(1)

    source_filename = Path(mp4_path).stem
    file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)

    if whisper_model:
        config.WHISPER_MODEL = whisper_model

    if not output_path:
        output_path = os.path.join(
            os.path.dirname(mp4_path),
            f"{source_filename} - Clean Cut.xml",
        )

    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline")
    print(f"{'='*60}")
    print(f"Source: {mp4_path}")
    print(f"Size: {file_size_mb:.0f} MB")

    result = _run_core(mp4_path, output_path, use_local)

    print(f"\n{'='*60}")
    print(f"Output: {result}")
    print(f"{'='*60}")
    print(f"\nNext: Open Premiere Pro > File > Import > select the .fcpxml")
    print(f"Premiere will ask you to relink media — point it to the original MP4.")
    return result


def run_notion(page_id: str, use_local: bool = False) -> str:
    """Notion + Drive mode — download from Drive, process, upload FCPXML, update Notion."""
    from drive import parse_drive_file_id, download_file, upload_file
    from notion_handler import get_page, update_xml_property

    output_folder_id = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
    if not output_folder_id:
        print("Error: GOOGLE_DRIVE_OUTPUT_FOLDER_ID not set in .env")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline — Notion Mode")
    print(f"{'='*60}")

    # Read Notion page
    print(f"\nReading Notion page {page_id}...")
    page = get_page(page_id)
    print(f"  Title: {page['title']}")

    if not page["raws_url"]:
        print("Error: No RAWs URL found on this Notion page")
        sys.exit(1)

    print(f"  RAWs URL: {page['raws_url']}")
    file_id = parse_drive_file_id(page["raws_url"])

    with tempfile.TemporaryDirectory(prefix="morningside_") as tmp_dir:
        # Download from Drive
        print("\nDownloading from Drive...")
        mp4_path = os.path.join(tmp_dir, "source.mp4")
        download_file(file_id, mp4_path)

        # Get original filename for the FCPXML
        from drive import get_file_name
        source_name = get_file_name(file_id)
        source_stem = Path(source_name).stem

        # Run core pipeline
        output_path = os.path.join(tmp_dir, f"{source_stem} - Clean Cut.xml")
        _run_core(mp4_path, output_path, use_local)

        # Upload FCPXML to Drive
        print("\nUploading FCPXML to Drive...")
        drive_url = upload_file(output_path, output_folder_id, f"{source_stem} - Clean Cut.xml")

    # Update Notion
    print("\nUpdating Notion...")
    update_xml_property(page_id, drive_url)

    print(f"\n{'='*60}")
    print(f"Complete! XML link written to Notion.")
    print(f"Drive URL: {drive_url}")
    print(f"{'='*60}")
    return drive_url


def run_watch(interval: int = 120, use_local: bool = False):
    """Watch mode — poll Notion DB every N seconds for new pages to process."""
    from notion_handler import get_ready_pages

    db_id = os.getenv("MORNINGSIDE_NOTION_DB_ID")
    if not db_id:
        print("Error: MORNINGSIDE_NOTION_DB_ID not set in .env")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Morningside XML Pipeline — Watch Mode")
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
    parser = argparse.ArgumentParser(description="Morningside XML Pipeline")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="Path to local MP4 file")
    mode.add_argument("--notion-id", help="Notion page ID to process")
    mode.add_argument("--watch", action="store_true", help="Poll Notion DB for new pages")

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
        run_local(args.file, args.out, args.whisper_model, use_local=args.local)
    elif args.notion_id:
        run_notion(args.notion_id, use_local=args.local)
    elif args.watch:
        run_watch(interval=args.interval, use_local=args.local)


if __name__ == "__main__":
    main()
