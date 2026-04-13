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


def _build_file_element(parent: ET.Element, source_filename: str, source_path: str | None,
                        timebase: int, ntsc: bool, total_frames: int,
                        width: int, height: int, sample_rate: int, audio_channels: int,
                        file_id: str = "file-1", define: bool = True) -> ET.Element:
    """Build a <file> element. If define=True, includes full metadata. Otherwise just a reference."""
    f_el = ET.SubElement(parent, "file", id=file_id)
    if not define:
        return f_el

    ET.SubElement(f_el, "name").text = source_filename
    if source_path:
        from urllib.parse import quote
        encoded = quote(source_path, safe="/:")
        ET.SubElement(f_el, "pathurl").text = f"file://localhost{encoded}"
    else:
        ET.SubElement(f_el, "pathurl").text = f"file://localhost/RELINK_ME/{source_filename}"
    _add_rate_element(f_el, timebase, ntsc)
    ET.SubElement(f_el, "duration").text = str(total_frames)

    f_media = ET.SubElement(f_el, "media")
    f_video = ET.SubElement(f_media, "video")
    f_vsc = ET.SubElement(f_video, "samplecharacteristics")
    _add_rate_element(f_vsc, timebase, ntsc)
    ET.SubElement(f_vsc, "width").text = str(width)
    ET.SubElement(f_vsc, "height").text = str(height)

    f_audio = ET.SubElement(f_media, "audio")
    f_asc = ET.SubElement(f_audio, "samplecharacteristics")
    ET.SubElement(f_asc, "depth").text = "16"
    ET.SubElement(f_asc, "samplerate").text = str(sample_rate)
    ET.SubElement(f_audio, "channelcount").text = str(audio_channels)

    return f_el


def _build_source_nest(parent: ET.Element, source_filename: str, source_path: str | None,
                       timebase: int, ntsc: bool, total_frames: int,
                       width: int, height: int, sample_rate: int, audio_channels: int,
                       nest_id: str = "source-nest") -> ET.Element:
    """Build a full-length nested sequence containing the source clip + empty track for external audio.

    The editor drops external mic audio into this nest. Cuts in the main
    timeline reference this nest, so both audio sources stay synced.
    """
    nest = ET.SubElement(parent, "sequence", id=nest_id)
    ET.SubElement(nest, "name").text = f"NEST - {source_filename}"
    ET.SubElement(nest, "duration").text = str(total_frames)
    _add_rate_element(nest, timebase, ntsc)

    tc = ET.SubElement(nest, "timecode")
    _add_rate_element(tc, timebase, ntsc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    n_media = ET.SubElement(nest, "media")

    # Video track — full source clip
    n_video = ET.SubElement(n_media, "video")
    n_vfmt = ET.SubElement(n_video, "format")
    n_vsc = ET.SubElement(n_vfmt, "samplecharacteristics")
    _add_rate_element(n_vsc, timebase, ntsc)
    ET.SubElement(n_vsc, "width").text = str(width)
    ET.SubElement(n_vsc, "height").text = str(height)
    ET.SubElement(n_vsc, "anamorphic").text = "FALSE"
    ET.SubElement(n_vsc, "pixelaspectratio").text = "square"
    ET.SubElement(n_vsc, "fielddominance").text = "none"

    n_vtrack = ET.SubElement(n_video, "track")
    v_clip = ET.SubElement(n_vtrack, "clipitem", id="nest-video-1")
    ET.SubElement(v_clip, "name").text = source_filename
    ET.SubElement(v_clip, "duration").text = str(total_frames)
    _add_rate_element(v_clip, timebase, ntsc)
    ET.SubElement(v_clip, "in").text = "0"
    ET.SubElement(v_clip, "out").text = str(total_frames)
    ET.SubElement(v_clip, "start").text = "0"
    ET.SubElement(v_clip, "end").text = str(total_frames)
    _build_file_element(v_clip, source_filename, source_path,
                        timebase, ntsc, total_frames, width, height,
                        sample_rate, audio_channels, file_id="file-1", define=True)

    # Audio tracks — camera audio (full clip)
    n_audio = ET.SubElement(n_media, "audio")
    n_afmt = ET.SubElement(n_audio, "format")
    n_asc = ET.SubElement(n_afmt, "samplecharacteristics")
    ET.SubElement(n_asc, "depth").text = "16"
    ET.SubElement(n_asc, "samplerate").text = str(sample_rate)

    for ch in range(audio_channels):
        n_atrack = ET.SubElement(n_audio, "track")
        a_clip = ET.SubElement(n_atrack, "clipitem", id=f"nest-audio-{ch+1}")
        ET.SubElement(a_clip, "name").text = source_filename
        ET.SubElement(a_clip, "duration").text = str(total_frames)
        _add_rate_element(a_clip, timebase, ntsc)
        ET.SubElement(a_clip, "in").text = "0"
        ET.SubElement(a_clip, "out").text = str(total_frames)
        ET.SubElement(a_clip, "start").text = "0"
        ET.SubElement(a_clip, "end").text = str(total_frames)
        _build_file_element(a_clip, source_filename, source_path,
                            timebase, ntsc, total_frames, width, height,
                            sample_rate, audio_channels, file_id="file-1", define=False)
        sourcetrack = ET.SubElement(a_clip, "sourcetrack")
        ET.SubElement(sourcetrack, "mediatype").text = "audio"
        ET.SubElement(sourcetrack, "trackindex").text = str(ch + 1)

    # Empty track for external mic — editor drops lav audio here
    ext_track = ET.SubElement(n_audio, "track")
    ET.SubElement(ext_track, "enabled").text = "TRUE"

    return nest


def generate_fcpxml(
    segments: list[dict],
    metadata: dict,
    source_filename: str,
    output_path: str,
    source_path: str | None = None,
) -> str:
    """Generate FCP 7 XML (Premiere Pro compatible) from segments and video metadata.

    Output contains two sequences:
    1. "NEST - filename" — full-length source clip with empty audio track for external mic
    2. "Clean Cut - filename" — the cut timeline referencing the nest

    Workflow: drop external mic audio into the nest, cuts propagate automatically.
    """
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    total_duration = metadata["duration_seconds"]
    audio_channels = metadata.get("audio_channels", 2)
    sample_rate = metadata.get("sample_rate", 48000)

    matched_fps, timebase = _match_fps(fps)
    ntsc = _is_ntsc(matched_fps)
    total_frames = _seconds_to_frames(total_duration, matched_fps)

    # Root — single sequence directly under xmeml.
    # Premiere only reliably reads the first <sequence> child.
    # The nest is omitted; editor can create it manually if needed.
    xmeml = ET.Element("xmeml", version="5")

    # --- Cut Sequence (references source file directly) ---
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = f"{source_filename} - Clean Cut"
    ET.SubElement(sequence, "duration").text = str(
        sum(_seconds_to_frames(s["end"] - s["start"], matched_fps) for s in segments)
    )
    _add_rate_element(sequence, timebase, ntsc)

    tc = ET.SubElement(sequence, "timecode")
    _add_rate_element(tc, timebase, ntsc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    media = ET.SubElement(sequence, "media")

    # Video track — clipitems reference the nest
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

    # Audio tracks for the cut sequence
    audio = ET.SubElement(media, "audio")
    a_fmt = ET.SubElement(audio, "format")
    a_sc = ET.SubElement(a_fmt, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text = "16"
    ET.SubElement(a_sc, "samplerate").text = str(sample_rate)

    # Camera audio channels + external mic channel from nest
    total_audio_tracks = audio_channels + 1
    a_tracks = []
    for ch in range(total_audio_tracks):
        a_track = ET.SubElement(audio, "track")
        a_tracks.append(a_track)

    nest_defined = False
    cumulative_frame = 0

    for i, seg in enumerate(segments):
        seg_start_frame = _seconds_to_frames(seg["start"], matched_fps)
        seg_duration_frames = _seconds_to_frames(seg["end"] - seg["start"], matched_fps)
        seg_end_frame = seg_start_frame + seg_duration_frames

        clip_id = f"clip-{i+1}"

        # --- Video clipitem referencing the nest ---
        v_clip = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(v_clip, "name").text = seg["label"]
        ET.SubElement(v_clip, "duration").text = str(total_frames)
        _add_rate_element(v_clip, timebase, ntsc)
        ET.SubElement(v_clip, "in").text = str(seg_start_frame)
        ET.SubElement(v_clip, "out").text = str(seg_end_frame)
        ET.SubElement(v_clip, "start").text = str(cumulative_frame)
        ET.SubElement(v_clip, "end").text = str(cumulative_frame + seg_duration_frames)

        # Reference the source file directly (not the nest — Premiere
        # needs a <file> reference to resolve clip media properly)
        _build_file_element(v_clip, source_filename, source_path,
                            timebase, ntsc, total_frames, width, height,
                            sample_rate, audio_channels,
                            file_id="file-1", define=(i == 0))

        # --- Audio clipitems referencing the source file ---
        for ch, a_track in enumerate(a_tracks):
            a_clip = ET.SubElement(a_track, "clipitem", id=f"{clip_id}-audio-{ch+1}")
            ET.SubElement(a_clip, "name").text = seg["label"]
            ET.SubElement(a_clip, "duration").text = str(total_frames)
            _add_rate_element(a_clip, timebase, ntsc)
            ET.SubElement(a_clip, "in").text = str(seg_start_frame)
            ET.SubElement(a_clip, "out").text = str(seg_end_frame)
            ET.SubElement(a_clip, "start").text = str(cumulative_frame)
            ET.SubElement(a_clip, "end").text = str(cumulative_frame + seg_duration_frames)
            _build_file_element(a_clip, source_filename, source_path,
                                timebase, ntsc, total_frames, width, height,
                                sample_rate, audio_channels,
                                file_id="file-1", define=False)

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
        '<!-- Two sequences will appear: -->\n'
        '<!--   NEST - contains full source clip + empty track for external mic -->\n'
        '<!--   Clean Cut - the cut timeline referencing the nest -->\n'
        '<!-- Drop external mic audio into the NEST sequence, cuts propagate automatically -->\n'
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
