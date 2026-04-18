"""
AttentionX - Clip Extractor
Uses MoviePy to:
  1. Trim the video to the target time window
  2. Apply the 9:16 smart-crop (face-centred)
  3. Burn karaoke captions as text overlays
  4. Export an optimised MP4 ready for TikTok / Reels / Shorts

Install: pip install moviepy
Also requires ffmpeg to be installed and on PATH.
"""

from pathlib import Path
from dataclasses import dataclass

# pip install moviepy
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.crop import crop as moviepy_crop


class ClipExtractor:
    """
    Takes a source video, a time window, crop parameters, and caption events,
    and writes a finished vertical short-form video clip to disk.
    """

    # ── Export settings (optimised for TikTok / Reels / Shorts) ──────────────
    OUTPUT_FPS = 30
    OUTPUT_BITRATE = "4000k"
    OUTPUT_AUDIO_BITRATE = "192k"
    OUTPUT_RESOLUTION = (1080, 1920)   # 9:16 @ 1080p

    # ── Caption styling ───────────────────────────────────────────────────────
    CAPTION_FONT = "DejaVu-Sans-Bold"   # any system font name or .ttf path
    CAPTION_FONTSIZE = 52
    CAPTION_COLOR = "white"
    CAPTION_STROKE_COLOR = "black"
    CAPTION_STROKE_WIDTH = 3
    CAPTION_Y_FRAC = 0.75              # vertical position (0 = top, 1 = bottom)

    HEADLINE_COLOR = "#E8FF47"         # AttentionX accent yellow
    HEADLINE_FONTSIZE = 44
    HEADLINE_Y_FRAC = 0.08
    HEADLINE_DURATION = 3.0            # show headline for first N seconds

    # ─────────────────────────────────────────────────────────────────────────
    #  Primary API
    # ─────────────────────────────────────────────────────────────────────────

    def extract_clip(
        self,
        input_path: str,
        output_path: str,
        start: float,
        end: float,
        crop_info=None,
        captions: list[dict] | None = None,
        hook_headline: str = "",
        enable_smart_crop: bool = True,
        enable_captions: bool = True,
        enable_headline: bool = True,
    ) -> str:
        """
        Full export pipeline: trim → crop → captions → headline → render.
        Returns the output_path on success.
        """
        print(f"[MoviePy] Extracting {Path(output_path).name} "
              f"({start:.1f}s – {end:.1f}s)...")

        # 1. Load and trim
        base = VideoFileClip(input_path).subclip(start, end)

        # 2. Smart crop to 9:16
        if enable_smart_crop and crop_info is not None:
            base = moviepy_crop(
                base,
                x1=crop_info.x,
                y1=crop_info.y,
                x2=crop_info.x + crop_info.width,
                y2=crop_info.y + crop_info.height,
            )
            base = base.resize(self.OUTPUT_RESOLUTION)
        else:
            # Centre-crop without face tracking
            base = self._centre_crop(base)

        layers = [base]

        # 3. Karaoke captions
        if enable_captions and captions:
            caption_clips = self._build_caption_clips(captions, base)
            layers.extend(caption_clips)

        # 4. Hook headline (first N seconds)
        if enable_headline and hook_headline:
            headline_clip = self._build_headline_clip(hook_headline, base)
            if headline_clip:
                layers.append(headline_clip)

        # 5. Composite and export
        final = CompositeVideoClip(layers)
        final.write_videofile(
            output_path,
            fps=self.OUTPUT_FPS,
            bitrate=self.OUTPUT_BITRATE,
            audio_bitrate=self.OUTPUT_AUDIO_BITRATE,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            logger=None,   # suppress moviepy progress spam
        )
        base.close()
        final.close()
        print(f"[MoviePy] ✓ Saved {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _centre_crop(self, clip: VideoFileClip) -> VideoFileClip:
        """Crop to 9:16 from the horizontal centre without face tracking."""
        w, h = clip.size
        target_w = int(h * 9 / 16)
        target_w = min(target_w, w)
        x1 = (w - target_w) // 2
        cropped = moviepy_crop(clip, x1=x1, y1=0, x2=x1 + target_w, y2=h)
        return cropped.resize(self.OUTPUT_RESOLUTION)

    def _build_caption_clips(
        self,
        captions: list[dict],
        base: VideoFileClip,
    ) -> list[TextClip]:
        """Convert caption event dicts into positioned MoviePy TextClips."""
        w, h = base.size
        y_pos = int(h * self.CAPTION_Y_FRAC)
        clips = []

        for cap in captions:
            t_start = float(cap.get("t", 0))
            t_end = float(cap.get("end", t_start + 2))
            text = str(cap.get("text", "")).strip()
            if not text:
                continue

            t_end = min(t_end, base.duration)
            duration = max(0.1, t_end - t_start)

            txt = (
                TextClip(
                    text,
                    fontsize=self.CAPTION_FONTSIZE,
                    font=self.CAPTION_FONT,
                    color=self.CAPTION_COLOR,
                    stroke_color=self.CAPTION_STROKE_COLOR,
                    stroke_width=self.CAPTION_STROKE_WIDTH,
                    method="caption",
                    size=(w - 80, None),
                    align="center",
                )
                .set_start(t_start)
                .set_duration(duration)
                .set_position(("center", y_pos))
            )
            clips.append(txt)

        return clips

    def _build_headline_clip(
        self,
        headline: str,
        base: VideoFileClip,
    ) -> TextClip | None:
        """Add a bold hook headline at the top for the first few seconds."""
        if not headline.strip():
            return None

        w, h = base.size
        y_pos = int(h * self.HEADLINE_Y_FRAC)
        duration = min(self.HEADLINE_DURATION, base.duration)

        return (
            TextClip(
                headline.upper(),
                fontsize=self.HEADLINE_FONTSIZE,
                font=self.CAPTION_FONT,
                color=self.HEADLINE_COLOR,
                stroke_color="black",
                stroke_width=4,
                method="caption",
                size=(w - 60, None),
                align="center",
            )
            .set_start(0)
            .set_duration(duration)
            .set_position(("center", y_pos))
        )
