"""
AttentionX - Transcriber
Uses OpenAI Whisper (local) to convert video audio into timestamped segments.

Install: pip install openai-whisper
Models:  tiny | base | small | medium | large
         (larger = more accurate, requires more RAM/GPU)
"""

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start: float        # seconds from video start
    end: float
    text: str
    avg_logprob: float  # Whisper confidence proxy (higher = more confident)


class Transcriber:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None   # lazy-loaded on first call

    def _load_model(self):
        if self._model is None:
            import whisper
            print(f"[Whisper] Loading model '{self.model_size}'...")
            self._model = whisper.load_model(self.model_size)
        return self._model

    # ──────────────────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────────────────

    def transcribe(self, video_path: str) -> list[TranscriptSegment]:
        """
        Transcribe the audio track of a video file.
        Returns one TranscriptSegment per spoken phrase with start/end timestamps.
        """
        model = self._load_model()
        print(f"[Whisper] Transcribing {video_path} ...")
        result = model.transcribe(
            video_path,
            verbose=False,
            word_timestamps=False,
            fp16=False,   # set True if using NVIDIA GPU
        )
        segments = []
        for seg in result["segments"]:
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                avg_logprob=seg.get("avg_logprob", 0.0),
            ))
        print(f"[Whisper] Transcribed {len(segments)} segments")
        return segments

    # ──────────────────────────────────────────────────────────────────────────
    #  Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    def segments_to_text(self, segments: list[TranscriptSegment]) -> str:
        """Concatenate all segments into a single string."""
        return " ".join(s.text for s in segments)

    def get_text_in_range(
        self,
        segments: list[TranscriptSegment],
        start: float,
        end: float,
    ) -> str:
        """Return the spoken text that falls within [start, end] seconds."""
        in_range = [s for s in segments if s.start >= start and s.end <= end]
        return " ".join(s.text for s in in_range)

    def format_for_gemini(self, segments: list[TranscriptSegment]) -> str:
        """
        Produce a timestamped transcript ready for Gemini analysis.

        Format:
            [08:23] And I remember standing there, completely overwhelmed...
            [08:51] That's when I realized the only validation I ever needed...
        """
        lines = []
        for seg in segments:
            mm = int(seg.start // 60)
            ss = int(seg.start % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {seg.text}")
        return "\n".join(lines)
