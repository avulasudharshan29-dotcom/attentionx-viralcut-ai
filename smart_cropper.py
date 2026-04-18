"""
AttentionX - Smart Cropper
Uses MediaPipe Face Detection to track the speaker's face across a video segment
and compute the optimal 9:16 crop window to keep them centred on screen.

Install: pip install mediapipe opencv-python-headless
"""

import cv2
import numpy as np
from dataclasses import dataclass

import mediapipe as mp


@dataclass
class CropInfo:
    x: int           # top-left x of the crop rectangle
    y: int           # top-left y (always 0 — full height)
    width: int       # crop width (= height × 9/16)
    height: int      # full video height
    confidence: float  # fraction of sampled frames where a face was found


class SmartCropper:
    """
    Analyses a video segment to find where the speaker's face is most often
    located, then returns a CropInfo that centres the face in a 9:16 frame.
    """

    SAMPLES_PER_SECOND = 2       # how many frames to sample per second
    TARGET_ASPECT = 9 / 16       # TikTok / Reels / Shorts aspect ratio

    def __init__(self):
        self._face_detection = None   # lazy-loaded

    def _get_detector(self):
        if self._face_detection is None:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,              # model 1 = full-range (up to 5 m)
                min_detection_confidence=0.5,
            )
        return self._face_detection

    # ──────────────────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_segment(
        self, video_path: str, start_sec: float, end_sec: float
    ) -> CropInfo:
        """
        Sample frames in [start_sec, end_sec], run face detection, and return
        the median horizontal face position as a stable 9:16 crop window.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Width of a 9:16 crop at full height
        crop_w = min(int(orig_h * self.TARGET_ASPECT), orig_w)

        face_cx_list = []
        total_sampled = 0
        sample_interval = max(1, int(fps / self.SAMPLES_PER_SECOND))

        detector = self._get_detector()

        frame_idx = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            total_sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                # Use the most prominent (largest) detection
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                cx = (bbox.xmin + bbox.width / 2) * orig_w
                face_cx_list.append(cx)

            frame_idx += sample_interval

        cap.release()

        # Compute crop position from median face centre
        if face_cx_list:
            median_cx = float(np.median(face_cx_list))
        else:
            median_cx = orig_w / 2      # fallback: centre of frame

        # Clamp so the crop rectangle stays within the frame
        crop_x = int(median_cx - crop_w / 2)
        crop_x = max(0, min(crop_x, orig_w - crop_w))

        confidence = len(face_cx_list) / total_sampled if total_sampled > 0 else 0.0

        return CropInfo(
            x=crop_x,
            y=0,
            width=crop_w,
            height=orig_h,
            confidence=round(confidence, 2),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Helper
    # ──────────────────────────────────────────────────────────────────────────

    def get_ffmpeg_crop_filter(self, crop_info: CropInfo) -> str:
        """
        Returns a ready-to-use ffmpeg crop filter string.
        Example: "crop=405:720:138:0"
        """
        return (f"crop={crop_info.width}:{crop_info.height}"
                f":{crop_info.x}:{crop_info.y}")
