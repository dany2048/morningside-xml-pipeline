"""Audio extraction and chunking via ffmpeg/ffprobe."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from config import AUDIO_BITRATE, WHISPER_MAX_CHUNK_MB, CHUNK_OVERLAP_SECONDS


def get_video_metadata(mp4_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video" and not video_stream:
            video_stream = stream
        elif stream["codec_type"] == "audio" and not audio_stream:
            audio_stream = stream

    fps = 24.0
    if video_stream:
        r_frame_rate = video_stream.get("r_frame_rate", "24/1")
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0

    duration = float(data.get("format", {}).get("duration", 0))

    audio_channels = int(audio_stream.get("channels", 2)) if audio_stream else 2
    sample_rate = int(audio_stream.get("sample_rate", 48000)) if audio_stream else 48000

    return {
        "fps": round(fps, 3),
        "duration_seconds": duration,
        "width": int(video_stream.get("width", 1920)) if video_stream else 1920,
        "height": int(video_stream.get("height", 1080)) if video_stream else 1080,
        "codec_name": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else "unknown",
        "audio_channels": audio_channels,
        "sample_rate": sample_rate,
    }


def extract_audio(mp4_path: str, output_path: str) -> str:
    """Extract audio from MP4 as MP3."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", AUDIO_BITRATE,
        str(output_path),
    ]
    print(f"  Extracting audio to {output_path}...")
    subprocess.run(cmd, capture_output=True, check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Audio extracted: {size_mb:.1f} MB")
    return output_path


def chunk_audio(audio_path: str, chunk_dir: str, max_mb: int = WHISPER_MAX_CHUNK_MB) -> list[tuple[str, float]]:
    """Split audio into chunks under max_mb, with overlap for boundary safety.

    Returns list of (chunk_path, start_offset_seconds).
    """
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if file_size_mb <= max_mb:
        return [(audio_path, 0.0)]

    # Get audio duration
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(json.loads(result.stdout)["format"]["duration"])

    # Calculate chunk duration to stay under max_mb
    num_chunks = math.ceil(file_size_mb / max_mb)
    chunk_duration = duration / num_chunks

    chunks = []
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_chunks):
        start = max(0, i * chunk_duration - (CHUNK_OVERLAP_SECONDS if i > 0 else 0))
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(chunk_duration + CHUNK_OVERLAP_SECONDS),
            "-acodec", "libmp3lame",
            "-ab", AUDIO_BITRATE,
            chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # The offset for timestamp correction (not counting overlap)
        offset = i * chunk_duration
        chunks.append((chunk_path, offset))
        print(f"  Chunk {i+1}/{num_chunks}: {start:.1f}s - {start + chunk_duration + CHUNK_OVERLAP_SECONDS:.1f}s")

    return chunks
