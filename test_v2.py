"""Quick test: run processor_v2 (GPT-5.4) on cached C5296 Whisper words.
Skips audio extraction and transcription — uses existing data.
Each run produces a uniquely-named XML for tracking.

Usage:
    python test_v2.py                    # auto-increments: v2, v3, v4...
    python test_v2.py --tag fewshot      # custom tag: ...Clean Cut fewshot.xml
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from dotenv import load_dotenv

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parent.parent
load_dotenv(_script_dir / ".env", override=True)
load_dotenv(_workspace_root / ".env")

from processor_v2 import process
from xml_gen import generate_fcpxml

RLHF_DIR = _workspace_root / "outputs" / "rlhf"
WORDS_FILE = RLHF_DIR / "c5296_words.json"
SOURCE_MP4 = _workspace_root / "reference" / "test file for cutting project C5296 .MP4"
OUTPUT_DIR = _workspace_root / "reference"
OUTPUT_BASE = "test file for cutting project C5296  - Clean Cut"


def _next_output_path(tag: str | None) -> Path:
    """Generate a unique output XML path.

    With --tag: ...Clean Cut {tag}.xml
    Without:    ...Clean Cut v{N}.xml  (auto-increments)
    """
    if tag:
        return OUTPUT_DIR / f"{OUTPUT_BASE} {tag}.xml"

    # Find highest existing vN and increment
    existing = list(OUTPUT_DIR.glob(f"{OUTPUT_BASE} v*.xml"))
    max_v = 1  # v1 is the original
    for f in existing:
        m = re.search(r"v(\d+)\.xml$", f.name)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return OUTPUT_DIR / f"{OUTPUT_BASE} v{max_v + 1}.xml"


def main():
    # Parse args
    tag = None
    if "--tag" in sys.argv:
        idx = sys.argv.index("--tag")
        if idx + 1 < len(sys.argv):
            tag = sys.argv[idx + 1]
        else:
            print("ERROR: --tag requires a value")
            sys.exit(1)

    output_xml = _next_output_path(tag)

    print("=" * 60)
    print("GPT-5.4 Pipeline Test — Project C5296")
    print(f"Output: {output_xml.name}")
    print("=" * 60)

    # Load cached words
    print(f"\nLoading cached Whisper words from {WORDS_FILE}...")
    with open(WORDS_FILE) as f:
        words = json.load(f)
    total_duration = words[-1]["end"]
    print(f"  {len(words)} words, {total_duration:.0f}s ({total_duration/60:.1f} min)")

    # Video metadata (hardcoded from previous run — C5296 is 4K 29.97fps)
    metadata = {
        "fps": 29.97,
        "duration_seconds": total_duration,
        "width": 3840,
        "height": 2160,
        "codec_name": "h264",
        "audio_channels": 2,
        "sample_rate": 48000,
    }

    # Run GPT-5.4 processor
    print(f"\nRunning GPT-5.4 single-pass processing...")
    segments = process(words, total_duration)

    if not segments:
        print("ERROR: No segments survived processing")
        sys.exit(1)

    # Generate FCPXML
    print(f"\nGenerating FCPXML...")
    generate_fcpxml(
        segments, metadata,
        "test file for cutting project C5296 .MP4",
        str(output_xml),
        source_path=str(SOURCE_MP4),
    )

    print(f"\n{'='*60}")
    print(f"Output: {output_xml}")
    print(f"{'='*60}")

    # List all versions for comparison
    all_versions = sorted(OUTPUT_DIR.glob(f"{OUTPUT_BASE}*.xml"))
    if len(all_versions) > 1:
        print(f"\nAll versions ({len(all_versions)}):")
        for v in all_versions:
            print(f"  {v.name}")


if __name__ == "__main__":
    main()
