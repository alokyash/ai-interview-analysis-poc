"""
Transcription module - Faster-Whisper (free, runs locally, no API key needed).

Model sizes (speed vs accuracy trade-off):
  tiny / base  -> fastest, fine for quick POC testing
  small        -> good balance, recommended default
  medium / large-v3 -> best accuracy, slower, needs more RAM/VRAM
"""

from faster_whisper import WhisperModel

_MODEL = None
_MODEL_SIZE = None


def get_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
    """Loads the Whisper model once and caches it for reuse."""
    global _MODEL, _MODEL_SIZE
    if _MODEL is None or _MODEL_SIZE != model_size:
        print(f"Loading Faster-Whisper model '{model_size}' ({device}, {compute_type})...")
        _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
        _MODEL_SIZE = model_size
    return _MODEL


def transcribe_audio(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribes an audio/video file and returns the full text plus
    timestamped segments.

    Returns:
        {
          "language": "en",
          "text": "full transcript as one string",
          "segments": [{"start": 0.0, "end": 3.2, "text": "..."}, ...]
        }
    """
    model = get_model(model_size=model_size)

    # vad_filter=True skips silence automatically - same idea as the
    # VAD-based chunking described in the real-time architecture doc.
    segments_gen, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    segments = []
    full_text_parts = []
    for seg in segments_gen:
        text = seg.text.strip()
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        full_text_parts.append(text)

    return {
        "language": info.language,
        "text": " ".join(full_text_parts),
        "segments": segments,
    }
