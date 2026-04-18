"""
AttentionX - Peak Detector
Combines two signals to find the best clip moments:
  1. Google Gemini 1.5 Flash  → semantic/emotional peak detection
  2. Librosa                  → raw audio energy spike detection

Final clips are ranked by a weighted fusion of both scores.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# pip install google-generativeai librosa soundfile
import google.generativeai as genai
import librosa


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Peak:
    start: float
    end: float
    text: str
    viral_score: float
    emotion_label: str          # Inspiring | Vulnerable | Shocking | etc.
    audio_energy: float
    tags: list[str] = field(default_factory=list)
    crop_info: Optional[object] = None
    captions: Optional[list] = None
    hook_headline: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini prompt
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = """\
You are an expert viral short-form video strategist.

You will receive a timestamped transcript from a long-form video (lecture, podcast, or talk).

Identify 10-15 "golden nugget" moments that would perform best as 60-second clips
on TikTok, Instagram Reels, or YouTube Shorts.

For each moment return a JSON array with this exact shape:
[
  {
    "start": <float, seconds>,
    "end": <float, seconds>,
    "text": "<the spoken text in this window>",
    "viral_score": <float 0.0 to 1.0>,
    "emotion_label": "<one of: Inspiring | Vulnerable | Shocking | Funny | Educational | Energetic | Profound>",
    "tags": ["<tag1>", "<tag2>"],
    "reason": "<one sentence why this moment is viral>"
  }
]

Scoring criteria (weight equally):
- Emotional intensity     Does it evoke strong feeling?
- Quotability             Is it punchy and shareable as a standalone?
- Story arc               Tension + resolution within 60 seconds?
- Universal appeal        Resonates beyond the niche audience?
- Hook quality            Do the first 3 seconds stop the scroll?

IMPORTANT: Return ONLY the raw JSON array. No markdown fences, no explanation.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  PeakDetector
# ─────────────────────────────────────────────────────────────────────────────

class PeakDetector:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Get a free key at https://aistudio.google.com/"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    # ──────────────────────────────────────────────────────────────────────────
    #  Gemini: semantic / emotional peaks
    # ──────────────────────────────────────────────────────────────────────────

    def find_emotional_peaks(self, transcript_segments) -> list[dict]:
        """
        Send the full timestamped transcript to Gemini 1.5 Flash and ask it
        to identify the most viral-worthy moments.
        """
        lines = []
        for seg in transcript_segments:
            mm, ss = int(seg.start // 60), int(seg.start % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {seg.text}")
        formatted = "\n".join(lines)

        prompt = f"{GEMINI_SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{formatted}"

        print("[Gemini] Sending transcript for peak analysis...")
        response = self.model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            peaks = json.loads(raw.strip())
            print(f"[Gemini] Identified {len(peaks)} candidate peaks")
            return peaks
        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON parse error: {e}")
            print(f"[Gemini] Raw response snippet: {raw[:300]}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    #  Librosa: audio energy peaks
    # ──────────────────────────────────────────────────────────────────────────

    def find_audio_peaks(
        self,
        video_path: str,
        window_sec: float = 60.0,
        hop_sec: float = 5.0,
        top_n: int = 20,
    ) -> list[dict]:
        """
        Slide a window across the video audio track and compute RMS energy.
        High RMS → speaker is speaking with more passion/volume.

        Returns top_n non-overlapping high-energy windows.
        """
        print("[Librosa] Loading audio for energy analysis...")
        y, sr = librosa.load(video_path, sr=16000, mono=True)

        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        energies = []
        position = 0
        while position + window_samples <= len(y):
            chunk = y[position: position + window_samples]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            start_sec = position / sr
            energies.append({
                "start": start_sec,
                "end": start_sec + window_sec,
                "rms": rms,
            })
            position += hop_samples

        # Sort descending by energy
        energies.sort(key=lambda x: x["rms"], reverse=True)

        # Non-maximum suppression: remove overlapping windows
        selected = []
        for e in energies:
            overlaps = any(
                abs(e["start"] - s["start"]) < window_sec * 0.7
                for s in selected
            )
            if not overlaps:
                selected.append(e)
            if len(selected) >= top_n:
                break

        print(f"[Librosa] Found {len(selected)} high-energy windows")
        return selected

    # ──────────────────────────────────────────────────────────────────────────
    #  Merge & rank
    # ──────────────────────────────────────────────────────────────────────────

    def rank_peaks(
        self,
        emotional_peaks: list[dict],
        audio_peaks: list[dict],
        max_clips: int = 6,
        clip_duration: int = 60,
        threshold: float = 0.70,
    ) -> list[Peak]:
        """
        Fuse Gemini semantic scores with Librosa audio energy scores.
        Final score = 70% Gemini + 30% audio energy (both normalised 0-1).
        Filter by threshold and return the top max_clips peaks.
        """
        max_rms = max((ap["rms"] for ap in audio_peaks), default=1.0)

        def nearest_audio_energy(start: float) -> float:
            """Find RMS of the audio window closest to this start time."""
            if not audio_peaks:
                return 0.0
            closest = min(audio_peaks, key=lambda ap: abs(ap["start"] - start))
            if abs(closest["start"] - start) < clip_duration:
                return closest["rms"]
            return 0.0

        merged = []
        for ep in emotional_peaks:
            audio_rms = nearest_audio_energy(ep["start"])
            audio_score = audio_rms / max_rms if max_rms > 0 else 0.0
            combined = round(0.70 * float(ep.get("viral_score", 0)) + 0.30 * audio_score, 3)

            if combined < threshold:
                continue

            # Pad clip to target duration if shorter
            start = float(ep["start"])
            end = float(ep["end"])
            if (end - start) < clip_duration:
                end = start + clip_duration

            merged.append(Peak(
                start=start,
                end=end,
                text=ep.get("text", ""),
                viral_score=combined,
                emotion_label=ep.get("emotion_label", "Inspiring"),
                audio_energy=audio_rms,
                tags=ep.get("tags", []),
            ))

        merged.sort(key=lambda p: p.viral_score, reverse=True)
        selected = merged[:max_clips]
        print(f"[Ranker] {len(selected)} clips selected (threshold={threshold})")
        return selected
