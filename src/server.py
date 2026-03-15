"""
FastAPI server wrapping the WhisperX predictor.

Exposes:
  GET  /health      → {"status": "ok"}
  POST /transcribe  → {"segments": [...], "detected_language": "..."}

The whisperx model is loaded once at startup and kept in VRAM between requests
(model caching is implemented in predict.py via _load_whisper).
A threading lock ensures sequential GPU access when requests arrive concurrently.
"""

import base64
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import login, whoami
from pydantic import BaseModel

from predict import Predictor
from speaker_processing import (
    identify_speakers_on_segments,
    load_known_speakers_from_samples,
    relabel_speakers_by_avg_similarity,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Globals — all tunable via pod environment variables
# ---------------------------------------------------------------------------
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

# Transcription defaults (overridable per-request, but these are the server defaults)
DEFAULT_BATCH_SIZE              = int(os.environ.get("BATCH_SIZE", "16"))
DEFAULT_BEAM_SIZE               = int(os.environ.get("BEAM_SIZE", "8"))
DEFAULT_TEMPERATURE             = float(os.environ.get("TEMPERATURE", "0.0"))
DEFAULT_VAD_ONSET               = float(os.environ.get("VAD_ONSET", "0.5"))
DEFAULT_VAD_OFFSET              = float(os.environ.get("VAD_OFFSET", "0.363"))
DEFAULT_CONDITION_ON_PREV_TEXT  = os.environ.get("CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
DEFAULT_NO_SPEECH_THRESHOLD     = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.75"))

MODEL: Optional[Predictor] = None
_GPU_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    # Authenticate with HuggingFace (needed for pyannote at inference time)
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            user = whoami(token=HF_TOKEN)
            logger.info("Hugging Face authenticated as: %s", user["name"])
        except Exception:
            logger.warning("HuggingFace authentication failed — diarization may not work", exc_info=True)
    else:
        logger.warning("HF_TOKEN not set — diarization will be unavailable")

    logger.info(
        "Config — batch_size=%d beam_size=%d temperature=%.1f vad_onset=%.3f vad_offset=%.3f "
        "condition_on_previous_text=%s no_speech_threshold=%.2f",
        DEFAULT_BATCH_SIZE, DEFAULT_BEAM_SIZE, DEFAULT_TEMPERATURE,
        DEFAULT_VAD_ONSET, DEFAULT_VAD_OFFSET,
        DEFAULT_CONDITION_ON_PREV_TEXT, DEFAULT_NO_SPEECH_THRESHOLD,
    )
    logger.info("Warming up WhisperX model…")
    MODEL = Predictor()
    MODEL.setup()
    logger.info("WhisperX model ready.")

    yield


app = FastAPI(title="WhisperX API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SpeakerSample(BaseModel):
    name: str
    url: str


def _transcribe_request_defaults() -> dict:
    """Build Field defaults from env vars at import time."""
    return dict(
        batch_size=DEFAULT_BATCH_SIZE,
        temperature=DEFAULT_TEMPERATURE,
        beam_size=DEFAULT_BEAM_SIZE,
        vad_onset=DEFAULT_VAD_ONSET,
        vad_offset=DEFAULT_VAD_OFFSET,
        condition_on_previous_text=DEFAULT_CONDITION_ON_PREV_TEXT,
        no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
    )


class TranscribeRequest(BaseModel):
    audio_file: str                              # URL or base64-encoded audio
    language: Optional[str] = None              # None = auto-detect
    language_detection_min_prob: float = 0.0
    language_detection_max_tries: int = 5
    initial_prompt: Optional[str] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    temperature: float = DEFAULT_TEMPERATURE
    beam_size: int = DEFAULT_BEAM_SIZE
    vad_onset: float = DEFAULT_VAD_ONSET
    vad_offset: float = DEFAULT_VAD_OFFSET
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREV_TEXT
    no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD
    align_output: bool = False
    diarization: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    speaker_samples: list[SpeakerSample] = []
    debug: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_path = _resolve_audio(req.audio_file)
    try:
        # Optional: enroll speaker profiles before transcription
        embeddings = {}
        if req.speaker_samples:
            samples = [{"name": s.name, "url": s.url} for s in req.speaker_samples]
            try:
                embeddings = load_known_speakers_from_samples(
                    samples, huggingface_access_token=HF_TOKEN
                )
                logger.info("Enrolled %d speaker profile(s).", len(embeddings))
            except Exception:
                logger.warning("Speaker profile enrollment failed, continuing without.", exc_info=True)

        # Run transcription — GPU-locked for sequential access
        with _GPU_LOCK:
            result = MODEL.predict(
                audio_file=audio_path,
                language=req.language,
                language_detection_min_prob=req.language_detection_min_prob,
                language_detection_max_tries=req.language_detection_max_tries,
                initial_prompt=req.initial_prompt,
                batch_size=req.batch_size,
                temperature=req.temperature,
                beam_size=req.beam_size,
                vad_onset=req.vad_onset,
                vad_offset=req.vad_offset,
                condition_on_previous_text=req.condition_on_previous_text,
                no_speech_threshold=req.no_speech_threshold,
                align_output=req.align_output,
                diarization=req.diarization,
                huggingface_access_token=HF_TOKEN,
                min_speakers=req.min_speakers,
                max_speakers=req.max_speakers,
                debug=req.debug,
            )

        segments = result.segments

        # Optional: re-label speakers using enrolled voice profiles
        if embeddings and segments:
            try:
                segments = identify_speakers_on_segments(
                    segments, str(audio_path), embeddings
                )
                segments = relabel_speakers_by_avg_similarity(segments)
            except Exception:
                logger.warning("Speaker identification failed, using raw labels.", exc_info=True)

        return JSONResponse(content={
            "segments": _serialize_segments(segments),
            "detected_language": result.detected_language,
        })

    finally:
        _cleanup(audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_audio(source: str) -> Path:
    """Download a URL or decode base64 audio into a named temp file."""
    if "://" in source:
        url_clean = source.split("?")[0]
        ext = Path(url_clean).suffix.lower()
        if ext not in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".webm", ".weba"}:
            ext = ".audio"
        try:
            resp = http_requests.get(source, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {exc}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(resp.content)
            return Path(f.name)
    else:
        if "," in source:
            source = source.split(",", 1)[1]
        try:
            data = base64.b64decode(source)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            f.write(data)
            return Path(f.name)


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _serialize_segments(segments: list) -> list:
    """Ensure all segment values are JSON-serializable (no numpy scalars)."""
    clean = []
    for seg in segments:
        clean.append({
            k: (float(v) if hasattr(v, "item") else v)
            for k, v in seg.items()
            if not k.startswith("__")
        })
    return clean
