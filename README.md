# AttentionX — Automated Content Repurposing Engine

Turn 60-minute lectures into 6 viral 60-second clips, automatically.

---

## Quick Start (VS Code)

### 1. Open the folder
File → Open Folder → select the `attentionx` folder

### 2. Create and activate a virtual environment
Open the integrated terminal (Ctrl + `)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Install ffmpeg (required by MoviePy):
- Windows: https://ffmpeg.org/download.html  (add to PATH)
- Mac:     brew install ffmpeg
- Ubuntu:  sudo apt install ffmpeg

### 4. Add your API key
```bash
cp .env.example .env
```
Open `.env` and paste your Gemini API key.
Get a free key at: https://aistudio.google.com/

### 5. Run the server
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /upload | Upload video, get job_id |
| POST | /process/{job_id} | Start AI pipeline |
| GET | /status/{job_id} | Poll progress & results |
| GET | /download/{job_id}/{file} | Download a clip MP4 |
| GET | /jobs | List all jobs |

### Example: full workflow with curl

```bash
# 1. Upload
curl -X POST http://localhost:8000/upload -F "file=@lecture.mp4"
# → {"job_id": "abc-123", ...}

# 2. Start processing
curl -X POST http://localhost:8000/process/abc-123 \
  -H "Content-Type: application/json" \
  -d '{
    "max_clips": 6,
    "clip_duration_seconds": 60,
    "viral_score_threshold": 0.70,
    "enable_smart_crop": true,
    "enable_captions": true,
    "enable_hook_headlines": true
  }'

# 3. Poll until status == "completed"
curl http://localhost:8000/status/abc-123

# 4. Download a clip
curl -O http://localhost:8000/download/abc-123/clip_01_score94.mp4
```

---

## Pipeline

```
Video Upload
    │
    ▼
Whisper (local)          — audio → timestamped transcript
    │
    ▼
Gemini 1.5 Flash         — find emotional peaks, score virality
    │
    ▼
Librosa                  — find audio energy spikes
    │
    ▼
Score Fusion             — 70% Gemini + 30% audio energy
    │
    ▼
MediaPipe                — face track → 9:16 crop info
    │
    ▼
Gemini 1.5 Flash         — karaoke captions + hook headline
    │
    ▼
MoviePy + ffmpeg         — trim, crop, burn captions, render MP4
    │
    ▼
Download-ready MP4 clips
```

---

## Project Structure

```
attentionx/
├── api/
│   └── main.py               FastAPI app + pipeline orchestration
├── core/
│   ├── transcriber.py        Whisper: audio → timestamped segments
│   ├── peak_detector.py      Gemini + Librosa peak detection & ranking
│   ├── smart_cropper.py      MediaPipe face tracking → 9:16 crop
│   ├── caption_generator.py  Gemini: karaoke captions + hook headlines
│   └── clip_extractor.py     MoviePy: trim, crop, captions, export
├── models/
│   └── schemas.py            Pydantic request/response models
├── uploads/                  Temp uploaded videos (auto-created)
├── outputs/                  Generated clips (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

| .env variable | Default | Description |
|---|---|---|
| GEMINI_API_KEY | (required) | Google AI Studio key |
| WHISPER_MODEL | base | tiny / base / small / medium / large |

| Process parameter | Default | Range |
|---|---|---|
| max_clips | 6 | 1–15 |
| clip_duration_seconds | 60 | 30–90 |
| viral_score_threshold | 0.70 | 0.0–1.0 |
| enable_smart_crop | true | — |
| enable_captions | true | — |
| enable_hook_headlines | true | — |

application demo video link:   https://drive.google.com/drive/folders/1oa-SrqF6WgpTK3sbjqcj3IS9lqPZ1aUB?usp=sharing

