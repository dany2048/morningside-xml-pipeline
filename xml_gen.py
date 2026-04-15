"""FCP 7 XML generation — Premiere Pro native import format.

Output structure:
  <xmeml>
    <sequence>                       # Clean Cut timeline
      <media>
        <video><track>
          <clipitem>                 # first cut — defines the nest inline
            <sequence id="nest-1">   # NEST — full source clip + empty mic track
              ...
            </sequence>
          </clipitem>
          <clipitem>                 # subsequent cuts — reference nest-1
            <sequence id="nest-1"/>
          </clipitem>
        </track></video>
        <audio>...same pattern...</audio>
      </media>
    </sequence>
  </xmeml>

Editor workflow: import → open "NEST - filename" → drop external mic audio →
cuts in "Clean Cut" inherit it automatically.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from config import FPS_TO_FRAME_DURATION

NEST_ID = "nest-1"
SOURCE_FILE_ID = "file-1"


def _match_fps(fps: float) -> tuple[float, int]:
    known = {
        23.976: 24, 24.0: 24, 25.0: 25, 29.97: 30,
        30.0: 30, 50.0: 50, 59.94: 60, 60.0: 60,
    }
    best_key = min(known.keys(), key=lambda k: abs(k - fps))
    return best_key, known[best_key]


def _is_ntsc(fps: float) -> bool:
    return fps in (23.976, 29.97, 59.94)


def _seconds_to_frames(seconds: float, fps: float) -> int:
    return round(seconds * fps)


def _add_rate(parent: ET.Element, timebase: int, ntsc: bool):
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _add_timecode(parent: ET.Element, timebase: int, ntsc: bool):
    tc = ET.SubElement(parent, "timecode")
    _add_rate(tc, timebase, ntsc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"


def _define_source_file(parent: ET.Element, source_filename: str, source_path: str | None,
                         timebase: int, ntsc: bool, total_frames: int,
                         width: int, height: int, sample_rate: int, audio_channels: int) -> ET.Element:
    f_el = ET.SubElement(parent, "file", id=SOURCE_FILE_ID)
    ET.SubElement(f_el, "name").text = source_filename
    if source_path:
        from urllib.parse import quote
        encoded = quote(source_path, safe="/:")
        ET.SubElement(f_el, "pathurl").text = f"file://localhost{encoded}"
    else:
        ET.SubElement(f_el, "pathurl").text = f"file://localhost/RELINK_ME/{source_filename}"
    _add_rate(f_el, timebase, ntsc)
    ET.SubElement(f_el, "duration").text = str(total_frames)
    _add_timecode(f_el, timebase, ntsc)

    f_media = ET.SubElement(f_el, "media")
    f_video = ET.SubElement(f_media, "video")
    f_vsc = ET.SubElement(f_video, "samplecharacteristics")
    _add_rate(f_vsc, timebase, ntsc)
    ET.SubElement(f_vsc, "width").text = str(width)
    ET.SubElement(f_vsc, "height").text = str(height)

    f_audio = ET.SubElement(f_media, "audio")
    f_asc = ET.SubElement(f_audio, "samplecharacteristics")
    ET.SubElement(f_asc, "depth").text = "16"
    ET.SubElement(f_asc, "samplerate").text = str(sample_rate)
    ET.SubElement(f_audio, "channelcount").text = str(audio_channels)
    return f_el


def _build_nest_sequence(parent: ET.Element, source_filename: str, source_path: str | None,
                          timebase: int, ntsc: bool, total_frames: int,
                          width: int, height: int, sample_rate: int, audio_channels: int) -> ET.Element:
    """Build the inline NEST sequence (full source + empty mic track). Defines source file inline."""
    nest = ET.SubElement(parent, "sequence", id=NEST_ID)
    ET.SubElement(nest, "name").text = f"NEST - {source_filename}"
    ET.SubElement(nest, "duration").text = str(total_frames)
    _add_rate(nest, timebase, ntsc)
    _add_timecode(nest, timebase, ntsc)

    n_media = ET.SubElement(nest, "media")

    # Video: full source clip
    n_video = ET.SubElement(n_media, "video")
    n_vfmt = ET.SubElement(n_video, "format")
    n_vsc = ET.SubElement(n_vfmt, "samplecharacteristics")
    _add_rate(n_vsc, timebase, ntsc)
    ET.SubElement(n_vsc, "width").text = str(width)
    ET.SubElement(n_vsc, "height").text = str(height)
    ET.SubElement(n_vsc, "anamorphic").text = "FALSE"
    ET.SubElement(n_vsc, "pixelaspectratio").text = "square"
    ET.SubElement(n_vsc, "fielddominance").text = "none"

    n_vtrack = ET.SubElement(n_video, "track")
    v_clip = ET.SubElement(n_vtrack, "clipitem", id="nest-video-1")
    ET.SubElement(v_clip, "name").text = source_filename
    ET.SubElement(v_clip, "duration").text = str(total_frames)
    _add_rate(v_clip, timebase, ntsc)
    ET.SubElement(v_clip, "in").text = "0"
    ET.SubElement(v_clip, "out").text = str(total_frames)
    ET.SubElement(v_clip, "start").text = "0"
    ET.SubElement(v_clip, "end").text = str(total_frames)
    _define_source_file(v_clip, source_filename, source_path,
                        timebase, ntsc, total_frames, width, height,
                        sample_rate, audio_channels)

    # Audio: camera channels (full clip) + one empty track for external mic
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
        _add_rate(a_clip, timebase, ntsc)
        ET.SubElement(a_clip, "in").text = "0"
        ET.SubElement(a_clip, "out").text = str(total_frames)
        ET.SubElement(a_clip, "start").text = "0"
        ET.SubElement(a_clip, "end").text = str(total_frames)
        ET.SubElement(a_clip, "file", id=SOURCE_FILE_ID)  # reference only
        sourcetrack = ET.SubElement(a_clip, "sourcetrack")
        ET.SubElement(sourcetrack, "mediatype").text = "audio"
        ET.SubElement(sourcetrack, "trackindex").text = str(ch + 1)

    # Empty track for external mic
    ext_track = ET.SubElement(n_audio, "track")
    ET.SubElement(ext_track, "enabled").text = "TRUE"
    ET.SubElement(ext_track, "locked").text = "FALSE"

    return nest


def generate_fcpxml(
    segments: list[dict],
    metadata: dict,
    source_filename: str,
    output_path: str,
    source_path: str | None = None,
) -> str:
    """Generate FCP 7 XML with inline-nested source sequence.

    The Clean Cut timeline's clipitems reference an inline NEST sequence
    (full source + empty mic track). Editor drops mic audio into the nest;
    cuts inherit automatically.
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

    xmeml = ET.Element("xmeml", version="5")

    # --- Clean Cut sequence (top level) ---
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = f"{source_filename} - Clean Cut"
    cut_total = sum(_seconds_to_frames(s["end"] - s["start"], matched_fps) for s in segments)
    ET.SubElement(sequence, "duration").text = str(cut_total)
    _add_rate(sequence, timebase, ntsc)
    _add_timecode(sequence, timebase, ntsc)

    media = ET.SubElement(sequence, "media")

    # Video track
    video = ET.SubElement(media, "video")
    vfmt = ET.SubElement(video, "format")
    vsc = ET.SubElement(vfmt, "samplecharacteristics")
    _add_rate(vsc, timebase, ntsc)
    ET.SubElement(vsc, "width").text = str(width)
    ET.SubElement(vsc, "height").text = str(height)
    ET.SubElement(vsc, "anamorphic").text = "FALSE"
    ET.SubElement(vsc, "pixelaspectratio").text = "square"
    ET.SubElement(vsc, "fielddominance").text = "none"

    v_track = ET.SubElement(video, "track")

    # Audio tracks: camera channels + external mic (mirror nest layout)
    audio = ET.SubElement(media, "audio")
    afmt = ET.SubElement(audio, "format")
    asc = ET.SubElement(afmt, "samplecharacteristics")
    ET.SubElement(asc, "depth").text = "16"
    ET.SubElement(asc, "samplerate").text = str(sample_rate)

    total_audio_tracks = audio_channels + 1  # +1 for external mic slot from nest
    a_tracks = [ET.SubElement(audio, "track") for _ in range(total_audio_tracks)]

    cumulative_frame = 0
    nest_defined = False

    for i, seg in enumerate(segments):
        seg_in = _seconds_to_frames(seg["start"], matched_fps)
        seg_dur = _seconds_to_frames(seg["end"] - seg["start"], matched_fps)
        seg_out = seg_in + seg_dur
        clip_id = f"clip-{i+1}"

        # --- Video clipitem — references the nest ---
        v_clip = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(v_clip, "name").text = seg["label"]
        ET.SubElement(v_clip, "duration").text = str(total_frames)
        _add_rate(v_clip, timebase, ntsc)
        ET.SubElement(v_clip, "in").text = str(seg_in)
        ET.SubElement(v_clip, "out").text = str(seg_out)
        ET.SubElement(v_clip, "start").text = str(cumulative_frame)
        ET.SubElement(v_clip, "end").text = str(cumulative_frame + seg_dur)

        if not nest_defined:
            _build_nest_sequence(v_clip, source_filename, source_path,
                                 timebase, ntsc, total_frames,
                                 width, height, sample_rate, audio_channels)
            nest_defined = True
        else:
            ET.SubElement(v_clip, "sequence", id=NEST_ID)

        # --- Audio clipitems — reference same nest, one per track ---
        for ch, a_track in enumerate(a_tracks):
            a_clip = ET.SubElement(a_track, "clipitem", id=f"{clip_id}-audio-{ch+1}")
            ET.SubElement(a_clip, "name").text = seg["label"]
            ET.SubElement(a_clip, "duration").text = str(total_frames)
            _add_rate(a_clip, timebase, ntsc)
            ET.SubElement(a_clip, "in").text = str(seg_in)
            ET.SubElement(a_clip, "out").text = str(seg_out)
            ET.SubElement(a_clip, "start").text = str(cumulative_frame)
            ET.SubElement(a_clip, "end").text = str(cumulative_frame + seg_dur)
            ET.SubElement(a_clip, "sequence", id=NEST_ID)
            sourcetrack = ET.SubElement(a_clip, "sourcetrack")
            ET.SubElement(sourcetrack, "mediatype").text = "audio"
            ET.SubElement(sourcetrack, "trackindex").text = str(ch + 1)

        cumulative_frame += seg_dur

    # Pretty print
    rough_string = ET.tostring(xmeml, encoding="unicode")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    comment = (
        '<!-- Generated by Morningside XML Pipeline -->\n'
        '<!-- Import into Premiere: File > Import > select this .xml -->\n'
        '<!-- Two sequences appear: Clean Cut + NEST - filename -->\n'
        '<!-- Drop external mic audio into the NEST; cuts inherit automatically -->\n'
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
