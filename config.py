"""Configuration constants for the Morningside XML pipeline."""

# Filler words to remove (case-insensitive)
FILLER_WORDS = {
    "um", "uh", "uhm", "uhh", "umm",
    "er", "err", "ah", "ahh",
    "like",  # only when filler (handled by context check)
    "basically", "literally", "actually", "honestly",
    "right", "okay", "so",  # only sentence-initial fillers
}

# Multi-word filler phrases
FILLER_PHRASES = [
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "you know what i mean",
]

# Repeat take detection
REPEAT_SIMILARITY_THRESHOLD = 0.65  # difflib ratio above this = same take
INCOMPLETE_DURATION_RATIO = 0.7     # below this % of longest take = incomplete

# Pause thresholds (seconds)
PAUSE_BETWEEN_UTTERANCES = 0.4  # gap > this starts a new utterance
REMOVABLE_PAUSE = 0.2           # gaps > this within a keeper get removed

# Segment assembly
SEGMENT_PADDING_SECONDS = 0.05  # 50ms padding on each side of kept segments

# Audio extraction
AUDIO_BITRATE = "128k"
WHISPER_MAX_CHUNK_MB = 24
CHUNK_OVERLAP_SECONDS = 2

# Common frame rates and their FCPXML rational representations
FPS_TO_FRAME_DURATION = {
    23.976: "1001/24000s",
    24.0:   "100/2400s",
    25.0:   "100/2500s",
    29.97:  "1001/30000s",
    30.0:   "100/3000s",
    50.0:   "100/5000s",
    59.94:  "1001/60000s",
    60.0:   "100/6000s",
}

# Whisper model for local transcription (openai-whisper)
# Options: "tiny", "base", "small", "medium", "large"
# "base" is fast and decent; "medium" is good balance; "large" is best but slow
WHISPER_MODEL = "base"

