import base64
import gc
import os
import sys
import shutil
import logging
import tempfile

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, whoami
import torch
import numpy as np
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

import predict as predict_module
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
from speaker_profiles import load_embeddings, relabel
from speaker_processing import (
    process_diarized_output, enroll_profiles, identify_speakers_on_segments,
    load_known_speakers_from_samples, identify_speaker, relabel_speakers_by_avg_similarity,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_' prefix - token may be malformed")

if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")




MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- 1) resolve audio input (URL or base64) -----------
    audio_input = job_input["audio_file"]
    try:
        if "://" in audio_input:
            # Standard URL — download as before
            audio_file_path = download_files_from_urls(job_id, [audio_input])[0]
            logger.debug(f"Audio downloaded → {audio_file_path}")
        else:
            # Treat as base64-encoded audio data
            # Strip optional data-URI prefix (e.g. "data:audio/wav;base64,")
            if "," in audio_input:
                audio_input = audio_input.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_input)
            os.makedirs(f"/jobs/{job_id}", exist_ok=True)
            audio_file_path = f"/jobs/{job_id}/audio_input"
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            logger.debug(f"Audio decoded from base64 → {audio_file_path} ({len(audio_bytes)} bytes)")
    except Exception as e:
        logger.error("Audio input failed", exc_info=True)
        return {"error": f"audio input: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token  # or job_input.get("huggingface_access_token")
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            embeddings = {}  # graceful degradation: proceed without profiles

    # ------------- 3) call WhisperX / VAD / diarization -------------
    # `diarization_turns`: when provided (chunked-whisper flow), the worker
    # skips its internal pyannote call and uses these turns directly. Saves
    # VRAM + time and lets every chunk share episode-stable speaker IDs.
    # `audio_offset_sec`: when > 0, every timestamp in the output is rebased
    # into the episode's absolute time frame before return.
    diarization_turns = job_input.get("diarization_turns") or None
    audio_offset_sec = float(job_input.get("audio_offset_sec", 0.0) or 0.0)

    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 16),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : job_input.get("huggingface_access_token") or hf_token,
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
        "diarization_turns"        : diarization_turns,
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        # Evict cached model + flush CUDA allocator so next job on same worker
        # starts with clean VRAM. Without this, an OOM mid-transcribe leaves
        # fragmented allocations that fail every subsequent job.
        cache = getattr(predict_module, "_whisper_cache", None)
        if cache and cache.get("model") is not None:
            cache["model"] = None
            cache["key"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language
    }
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    # Threshold controls how close the ECAPA cosine similarity must be before
    # a diarized SPEAKER_XX label is auto-replaced with an enrolled name.
    # 0.1 (old default) was effectively "always match", 0.85 is the recurring-
    # host threshold from the pipeline plan. tenant-backend sets it via the
    # speaker_match_threshold job input; this stays configurable from outside.
    speaker_match_threshold = float(job_input.get("speaker_match_threshold", 0.85))
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=speaker_match_threshold,
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully (threshold=%s).", speaker_match_threshold)
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # 5) Strip segments to essential fields, merge adjacent same-speaker segments,
    #    then gzip+base64 compress to stay within RunPod's payload limit.
    import gzip as _gzip
    import json as _json

    keep_words = job_input.get("align_output", False)

    # 5a) Strip to minimal fields + preserve avg_logprob (faster-whisper gives
    # it per segment) as a normalised confidence in [0, 1]. Exp(avg_logprob)
    # is the model's probability estimate for the transcribed text.
    def _confidence(seg):
        lp = seg.get("avg_logprob")
        if lp is None:
            return None
        try:
            import math as _math
            return round(_math.exp(lp), 4)
        except Exception:
            return None

    minimal_segments = []
    for seg in output_dict.get("segments", []):
        clean = {
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
        }
        if seg.get("speaker"):
            clean["speaker"] = seg["speaker"]
        conf = _confidence(seg)
        if conf is not None:
            clean["confidence"] = conf
        if keep_words and seg.get("words"):
            # Pass through per-word score when whisperx emits it. Used by
            # the downstream transcript-correction stage to identify
            # low-confidence proper-noun spans; falls back gracefully on
            # segment-level confidence when missing.
            clean["words"] = [
                {
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "word": w.get("word", ""),
                    "score": w.get("score", w.get("probability")),
                }
                for w in seg["words"]
            ]
        if clean["text"]:
            minimal_segments.append(clean)

    # 5b) Merge adjacent segments with the same speaker; take a
    # duration-weighted average of the per-segment confidences.
    def _merge_conf(a, b, a_dur, b_dur):
        ca = a.get("confidence")
        cb = b.get("confidence")
        if ca is None and cb is None:
            return None
        if ca is None:
            return cb
        if cb is None:
            return ca
        total = a_dur + b_dur
        if total <= 0:
            return (ca + cb) / 2
        return round((ca * a_dur + cb * b_dur) / total, 4)

    merged = []
    for seg in minimal_segments:
        if merged and merged[-1].get("speaker") == seg.get("speaker") and seg.get("speaker"):
            prev = merged[-1]
            prev_dur = prev["end"] - prev["start"]
            seg_dur = seg["end"] - seg["start"]
            mc = _merge_conf(prev, seg, prev_dur, seg_dur)
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
            if mc is not None:
                prev["confidence"] = mc
        else:
            merged.append(seg)

    # 5b.5) Rebase timestamps into the episode's absolute time frame when
    # this job was a chunk of a larger audio file. The caller passes the
    # chunk's offset via `audio_offset_sec`; we add it to every segment's
    # and word's start/end so the caller can stitch N chunk outputs with
    # monotonic timestamps without any post-processing on its side.
    if audio_offset_sec > 0:
        for seg in merged:
            seg["start"] = round(seg.get("start", 0) + audio_offset_sec, 3)
            seg["end"] = round(seg.get("end", 0) + audio_offset_sec, 3)
            for w in seg.get("words", []) or []:
                if w.get("start") is not None:
                    w["start"] = round(float(w["start"]) + audio_offset_sec, 3)
                if w.get("end") is not None:
                    w["end"] = round(float(w["end"]) + audio_offset_sec, 3)
        logger.info(f"Rebased {len(merged)} segments by +{audio_offset_sec:.2f}s")

    logger.info(f"Segments: {len(minimal_segments)} raw → {len(merged)} after merge")

    # 5c) Check payload size — if over 10 MB, gzip+base64 compress the segments.
    #     The consumer (tenant-backend) detects "segments_gz" and decompresses.
    segments_json = _json.dumps(merged, ensure_ascii=False)
    raw_size = len(segments_json.encode('utf-8'))
    MAX_INLINE_BYTES = 10 * 1024 * 1024  # 10 MB

    if raw_size > MAX_INLINE_BYTES:
        compressed = base64.b64encode(_gzip.compress(segments_json.encode('utf-8'))).decode('ascii')
        logger.info(f"Segments payload {raw_size/1024/1024:.1f} MB → compressed to {len(compressed)/1024/1024:.1f} MB")
        output_dict.pop("segments", None)
        output_dict["segments_gz"] = compressed
        output_dict["segments_count"] = len(merged)
    else:
        output_dict["segments"] = merged

    # 6-Cleanup and return output_dict normally
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict

runpod.serverless.start({"handler": run})