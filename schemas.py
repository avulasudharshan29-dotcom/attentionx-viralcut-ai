"""
AttentionX - Pydantic Schemas
"""
from typing import Optional
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    max_clips: int = Field(
        default=6, ge=1, le=15,
        description="Maximum clips to generate")
    clip_duration_seconds: int = Field(
        default=60, ge=30, le=90,
        description="Target clip duration in seconds")
    viral_score_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Minimum viral score to include a clip (0.0 – 1.0)")
    enable_smart_crop: bool = Field(
        default=True, description="Enable 9:16 face-tracking crop")
    enable_captions: bool = Field(
        default=True, description="Burn karaoke captions into video")
    enable_hook_headlines: bool = Field(
        default=True, description="Generate AI hook headlines")


class ClipResult(BaseModel):
    filename: str
    download_url: str
    start_time: float
    end_time: float
    viral_score: float
    hook_headline: str
    tags: list[str]
    transcript_segment: str
    emotion_label: str = ""


class ProcessStatus(BaseModel):
    job_id: str
    status: str          # uploaded | processing | completed | failed
    clips: list[dict] = []
    error: Optional[str] = None
    progress: int = 0
    current_step: str = ""
