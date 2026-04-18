"""
AttentionX - Caption Generator
Uses Gemini 1.5 Flash to produce:
  1. Hook headlines  — punchy scroll-stopping titles for the clip
  2. Karaoke captions — timed caption events for burning into the video
"""

import os
import json
import google.generativeai as genai


# ─────────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────────

HOOK_PROMPT = """\
You are a viral short-form video copywriter specialising in educational content.

Write ONE hook headline for a clip with emotion label: {emotion}

Rules:
- Maximum 12 words
- Creates immediate curiosity or emotional resonance
- Never clickbait or misleading
- No hashtags
- Match the emotion label tone

TRANSCRIPT EXCERPT:
{text}

Return ONLY the headline text. No quotes, no punctuation at the end, no explanation.
"""

CAPTION_PROMPT = """\
You are a caption timing expert for short-form social video.

The clip starts at {start_sec:.1f} seconds into the original video and lasts {duration} seconds.

Create karaoke-style word-group captions for this transcript.
Group 3-5 words per caption line. Keep timing tight and natural.

Return a JSON array where each item is:
{{"t": <float, seconds from clip start>, "end": <float, seconds from clip start>, "text": "<caption text>"}}

All times must be relative to clip start (so first caption starts near 0.0).
Cover the full {duration} seconds.

TRANSCRIPT:
{text}

Return ONLY the raw JSON array. No markdown fences, no explanation.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CaptionGenerator
# ─────────────────────────────────────────────────────────────────────────────

class CaptionGenerator:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    # ──────────────────────────────────────────────────────────────────────────
    #  Hook headline
    # ──────────────────────────────────────────────────────────────────────────

    def generate_hook(self, text: str, peak: dict) -> str:
        """Generate a viral hook headline for a clip."""
        if not text.strip():
            return ""
        emotion = peak.get("emotion_label", "Inspiring")
        prompt = HOOK_PROMPT.format(emotion=emotion, text=text[:1000])
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[Caption] Hook generation failed: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    #  Karaoke captions
    # ──────────────────────────────────────────────────────────────────────────

    def generate_captions(self, text: str, peak: dict) -> list[dict]:
        """
        Generate timed karaoke caption events for a clip.
        Returns: [{"t": float, "end": float, "text": str}, ...]
        """
        if not text.strip():
            return []

        start = float(peak.get("start", 0))
        end = float(peak.get("end", 60))
        duration = int(end - start)

        prompt = CAPTION_PROMPT.format(
            start_sec=start,
            duration=duration,
            text=text[:2000],
        )

        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]

            captions = json.loads(raw.strip())
            print(f"[Caption] Generated {len(captions)} caption events")
            return captions

        except Exception as e:
            print(f"[Caption] Caption generation failed ({e}), using fallback")
            return self._fallback_captions(text, duration)

    # ──────────────────────────────────────────────────────────────────────────
    #  Fallback: simple word-chunk distributor
    # ──────────────────────────────────────────────────────────────────────────

    def _fallback_captions(self, text: str, duration: float) -> list[dict]:
        """
        If Gemini fails, distribute words evenly across the clip duration.
        Groups 4 words per caption line.
        """
        words = text.split()
        chunk_size = 4
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        if not chunks:
            return []

        time_per_chunk = duration / len(chunks)
        return [
            {
                "t": round(i * time_per_chunk, 2),
                "end": round((i + 1) * time_per_chunk, 2),
                "text": " ".join(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]
