"""FCP 7 XML generation — Premiere Pro native import format."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from config import FPS_TO_FRAME_DURATION


def _match_fps(fps: float) -> tuple[float, int]:
    """Match fps to nearest standard value. Returns (matched_fps, timebase)."""
    known = {
        23.976: 24,
        24.0: 24,
        25.0: 25,
        29.97: 30,
        30.0: 30,
        50.0: 50,
        59.94: 60,
        60.0: 60,
    }
    best_key = min(known.keys(), key=lambda k: abs(k - fps))
    return best_key, known[best_key]


def _is_ntsc(fps: float) -> bool:
    return fps in (23.976, 29.97, 59.94)


def _seconds_to_frames(seconds: float, fps: float) -> int:
    return round(seconds * fps)


def _add_rate_element(parent: ET.Element, timebase: int, ntsc: bool):
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def generate_fcpxml(
    segments: list[dict],
    metadata: dict,
    source_filename: str,
    output_path: str,
    source_path: str | None = None,
) -> str:
    """Generate FCP 7 XML (Premiere Pro compatible) from segments and video metadata."""
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    total_duration = metadata["duration_seconds"]
    audio_channels = metadata.get("audio_channels", 2)
    sample_rate = metadata.get("sample_rate", 48000)

    matched_fps, timebase = _match_fps(fps)
    ntsc = _is_ntsc(matched_fps)
    total_frames = _seconds_to_frames(total_duration, matched_fps)

    # Root
    xmeml = ET.Element("xmeml", version="5")

    # Sequence
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = f"{source_filename} - Clean Cut"
    ET.SubElement(sequence, "duration").text = str(
        sum(_seconds_to_frames(s["end"] - s["start"], matched_fps) for s in segments)
    )
    _add_rate_element(sequence, timebase, ntsc)

    # Timecode
    tc = ET.SubElement(sequence, "timecode")
    _add_rate_element(tc, timebase, ntsc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    # Media
    media = ET.SubElement(sequence, "media")

    # === Video ===
    video = ET.SubElement(media, "video")
    fmt = ET.SubElement(video, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    _add_rate_element(sc, timebase, ntsc)
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    ET.SubElement(sc, "anamorphic").text = "FALSE"
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    ET.SubElement(sc, "fielddominance").text = "none"

    v_track = ET.SubElement(video, "track")

    # === Audio ===
    audio = ET.SubElement(media, "audio")
    a_fmt = ET.SubElement(audio, "format")
    a_sc = ET.SubElement(a_fmt, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text = "16"
    ET.SubElement(a_sc, "samplerate").text = str(sample_rate)

    # Premiere expects one track per audio channel for stereo
    a_tracks = []
    for ch in range(audio_channels):
        a_track = ET.SubElement(audio, "track")
        a_tracks.append(a_track)

    # Build file element once (referenced by id afterwards)
    file_element_built = False

    cumulative_frame = 0
    for i, seg in enumerate(segments):
        seg_start_frame = _seconds_to_frames(seg["start"], matched_fps)
        seg_duration_frames = _seconds_to_frames(seg["end"] - seg["start"], matched_fps)
        seg_end_frame = seg_start_frame + seg_duration_frames

        clip_id = f"clip-{i+1}"

        # --- Video clipitem ---
        v_clip = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(v_clip, "name").text = seg["label"]
        ET.SubElement(v_clip, "duration").text = str(total_frames)
        _add_rate_element(v_clip, timebase, ntsc)
        ET.SubElement(v_clip, "in").text = str(seg_start_frame)
        ET.SubElement(v_clip, "out").text = str(seg_end_frame)
        ET.SubElement(v_clip, "start").text = str(cumulative_frame)
        ET.SubElement(v_clip, "end").text = str(cumulative_frame + seg_duration_frames)

        # File reference
        v_file = ET.SubElement(v_clip, "file", id="file-1")
        if not file_element_built:
            ET.SubElement(v_file, "name").text = source_filename
            if source_path:
                # Absolute path so Premiere auto-links the media
                from urllib.parse import quote
                encoded = quote(source_path, safe="/:")
                ET.SubElement(v_file, "pathurl").text = f"file://localhost{encoded}"
            else:
                ET.SubElement(v_file, "pathurl").text = f"file://localhost/RELINK_ME/{source_filename}"
            _add_rate_element(v_file, timebase, ntsc)
            ET.SubElement(v_file, "duration").text = str(total_frames)

            f_media = ET.SubElement(v_file, "media")

            # Video inside file
            f_video = ET.SubElement(f_media, "video")
            f_vsc = ET.SubElement(f_video, "samplecharacteristics")
            _add_rate_element(f_vsc, timebase, ntsc)
            ET.SubElement(f_vsc, "width").text = str(width)
            ET.SubElement(f_vsc, "height").text = str(height)

            # Audio inside file — must declare all channels
            f_audio = ET.SubElement(f_media, "audio")
            f_asc = ET.SubElement(f_audio, "samplecharacteristics")
            ET.SubElement(f_asc, "depth").text = "16"
            ET.SubElement(f_asc, "samplerate").text = str(sample_rate)
            # channelcount tells Premiere the source has stereo
            ET.SubElement(f_audio, "channelcount").text = str(audio_channels)

            file_element_built = True

        # --- Audio clipitems (one per channel) ---
        for ch, a_track in enumerate(a_tracks):
            a_clip = ET.SubElement(a_track, "clipitem", id=f"{clip_id}-audio-{ch+1}")
            ET.SubElement(a_clip, "name").text = seg["label"]
            ET.SubElement(a_clip, "duration").text = str(total_frames)
            _add_rate_element(a_clip, timebase, ntsc)
            ET.SubElement(a_clip, "in").text = str(seg_start_frame)
            ET.SubElement(a_clip, "out").text = str(seg_end_frame)
            ET.SubElement(a_clip, "start").text = str(cumulative_frame)
            ET.SubElement(a_clip, "end").text = str(cumulative_frame + seg_duration_frames)
            ET.SubElement(a_clip, "file", id="file-1")

            # sourcetrack tells Premiere which channel this clip maps to
            sourcetrack = ET.SubElement(a_clip, "sourcetrack")
            ET.SubElement(sourcetrack, "mediatype").text = "audio"
            ET.SubElement(sourcetrack, "trackindex").text = str(ch + 1)

        cumulative_frame += seg_duration_frames

    # Pretty print
    rough_string = ET.tostring(xmeml, encoding="unicode")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    comment = (
        '<!-- Generated by Morningside XML Pipeline -->\n'
        '<!-- Import into Premiere Pro: File > Import > select this .xml file -->\n'
        '<!-- Premiere will ask to relink media — point it to the original MP4 -->\n'
    )

    parsed = minidom.parseString(rough_string)
    pretty = parsed.toprettyxml(indent="  ")
    lines = pretty.split("\n")[1:]
    pretty_body = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_declaration)
        f.write(comment)
        f.write(pretty_body)

    print(f"  FCP7 XML written to {output_path}")
    return output_path
