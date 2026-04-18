import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# ✅ CHANGED HERE
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ✅ CHANGED HERE
from transcriber import Transcriber
from peak_detector import PeakDetector
from clip_extractor import ClipExtractor
from caption_generator import CaptionGenerator
from smart_cropper import SmartCropper
from schemas import ProcessRequest, ProcessStatus, ClipResult

load_dotenv()

app = FastAPI(
    title="AttentionX API",
    description="Automated short-form content repurposing engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

jobs = {}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "file_path": str(file_path),
        "clips": [],
        "error": None,
        "progress": 0,
        "current_step": "",
    }

    return {"job_id": job_id, "filename": file.filename}

@app.get("/jobs")
async def list_jobs():
    return jobs