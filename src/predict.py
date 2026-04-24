#from cog import BasePredictor, Input, Path, BaseModel
try:
    # Prefer real cog if present (e.g. when running locally)
    from cog import BasePredictor, Input, Path, BaseModel
except ImportError:                          # pragma: no cover
    from cog_stub import BasePredictor, Input, Path, BaseModel
from pydub import AudioSegment
from typing import Any
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from scipy.spatial.distance import cosine
import gc
import math
import os
import shutil
import whisperx
# whisperx 3.8.1 stopped auto-importing its .alignment submodule at package
# init, so `whisperx.alignment.DEFAULT_ALIGN_MODELS_*` raises AttributeError
# unless the submodule is explicitly imported first. Force-import here so
# the `if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_*`
# check below keeps working.
import whisperx.alignment  # noqa: F401 — imported for side-effect
import tempfile
import time
import torch
import speaker_processing


import logging
import sys
logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)






torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3"

# ---------------------------------------------------------------------------
# Module-level model cache — keeps the whisperx model in VRAM between calls
# so a persistent pod (FastAPI server) pays the load cost only once.
# ---------------------------------------------------------------------------
_whisper_cache: dict = {"model": None, "key": None}


def _load_whisper(language, asr_options: dict, vad_options: dict):
    """Return a cached FasterWhisperPipeline if options match, otherwise reload."""
    global _whisper_cache

    def _make_key(d: dict):
        return tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in d.items()
        ))

    key = (language, _make_key(asr_options), _make_key(vad_options))

    if _whisper_cache["model"] is None or _whisper_cache["key"] != key:
        if _whisper_cache["model"] is not None:
            del _whisper_cache["model"]
            torch.cuda.empty_cache()
        logger.info("Loading whisperx model (arch=%s, lang=%s)…", whisper_arch, language)
        _whisper_cache["model"] = whisperx.load_model(
            whisper_arch, device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
            vad_options=vad_options,
        )
        _whisper_cache["key"] = key
        logger.info("Whisperx model loaded and cached.")

    return _whisper_cache["model"]


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        # Ensure VAD model is in torch cache (fallback for whisperx)
        vad_cache = '/root/.cache/torch/whisperx-vad-segmentation.bin'
        if not os.path.exists(vad_cache):
            bundled = './models/vad/whisperx-vad-segmentation.bin'
            if os.path.exists(bundled):
                os.makedirs('/root/.cache/torch', exist_ok=True)
                shutil.copy(bundled, vad_cache)

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=16),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            beam_size: int = Input(
                description="Beam search width — higher is more accurate but slower",
                default=8),
            condition_on_previous_text: bool = Input(
                description="Feed previous output as prompt for next window; False prevents error cascade on long/mixed-language audio",
                default=False),
            no_speech_threshold: float = Input(
                description="Segments below this log-probability are treated as silence",
                default=0.75),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=False),
            speaker_verification: bool = Input(
                description="Enable speaker verification",
                default=False),
            speaker_samples: list = Input(
                description="List of speaker samples for verification. Each sample should be a dict with 'url' and "
                            "optional 'name' and 'file_path'. If 'name' is not provided, the file name (without "
                            "extension) is used. If 'file_path' is provided, it will be used directly.",
                default=[]
            ),
            diarization_turns: list = Input(
                description="Pre-computed pyannote diarization turns "
                            "[{start, end, speaker}, ...]. When provided, the worker SKIPS its internal "
                            "DiarizationPipeline call and uses these turns directly to assign word speakers. "
                            "Used by the chunked-transcription flow so every chunk shares episode-stable "
                            "speaker IDs assigned by a single pyannote pass over the full audio.",
                default=None
            )
    ) -> Output:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
                "beam_size": beam_size,
                "condition_on_previous_text": condition_on_previous_text,
                "no_speech_threshold": no_speech_threshold,
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            audio_duration = get_audio_duration(audio_file)

            if language is None and language_detection_min_prob > 0 and audio_duration > 30000:
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms)
                )

                segments_starts = distribute_segments_equally(audio_duration, segments_duration_ms,
                                                              language_detection_max_tries)

                logger.info("Detecting languages on segments starting at " + ', '.join(map(str, segments_starts)))

                detected_language_details = detect_language(audio_file, segments_starts, language_detection_min_prob,
                                                            language_detection_max_tries, asr_options, vad_options)

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                logger.info(f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                      f"{detected_language_iterations} iterations.")

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e6

            model = _load_whisper(language, asr_options, vad_options)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    result = align(audio, result, debug)
                else:
                    logger.warning(f"Cannot align output as language {detected_language} is not supported for alignment")

            # Speaker attribution is always driven by the Diarization
            # worker's pre-computed turns. The in-worker pyannote path
            # was removed — keeping two diarization sources produced
            # episode-unstable labels when the two models disagreed on
            # speaker count, and doubled VRAM for no accuracy gain.
            if diarization_turns:
                result = assign_speakers_from_turns(result, diarization_turns, debug)
            elif diarization:
                logger.warning(
                    "diarization=True but diarization_turns is empty; "
                    "skipping attribution. Caller must supply turns from "
                    "the Diarization worker."
                )

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    
    return len(AudioSegment.from_file(file_path))


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = _load_whisper(None, asr_options, vad_options)

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    logger.info(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path

    audio = AudioSegment.from_file(input_file_path)

    end_time_ms = start_time_ms + duration_ms
    extracted_segment = audio[start_time_ms:end_time_ms]

    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)
        extracted_segment.export(temp_file_path, format=file_extension.lstrip('.'))

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def assign_speakers_from_turns(result, turns, debug, boundary_split_min_sec=0.3):
    """Custom overlap-weighted speaker attribution.

    WhisperX's `assign_word_speakers` picks the speaker whose turn
    contains the *midpoint* of each word, which fails at turn boundaries
    (quick "mhm" overlapping the other speaker's sentence) and it shares
    the label across every word of a whisper segment even when the
    segment straddles a speaker change. Our custom pass instead:

      1. For each word, compute overlap duration with every turn speaker
         and assign the speaker with max overlap.
      2. If a word spans a turn boundary and each side gets
         ≥ `boundary_split_min_sec` of overlap, duplicate the word into
         two half-words so both speakers keep the text — callers can
         drop duplicates later, but we won't lose text across a
         boundary.
      3. Fall back to nearest turn by center-distance if the word has
         no overlap at all (shouldn't happen unless the word falls in
         a VAD-silence hole).
      4. Re-group words into speaker-contiguous segments so the output
         has one segment per speaker turn inside the transcript, never
         a mixed-speaker segment.

    Input turns: [{"start", "end", "speaker"}, ...]
    """
    start_time = time.time_ns() / 1e6

    norm_turns: list[tuple[float, float, str]] = []
    for t in turns or []:
        try:
            s = float(t["start"])
            e = float(t["end"])
            spk = str(t["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s and spk:
            norm_turns.append((s, e, spk))
    if not norm_turns:
        logger.warning("assign_speakers_from_turns: empty/malformed turns; leaving segments unlabelled")
        return result

    norm_turns.sort(key=lambda t: t[0])

    def _word_speaker(ws: float, we: float) -> list[tuple[str, float]]:
        """Return up to two (speaker, overlap_sec) pairs sorted by
        overlap desc. One entry for the clear-winner case, two for a
        boundary-straddling word that deserves to be split."""
        if we <= ws:
            return []
        overlaps: dict[str, float] = {}
        for s, e, spk in norm_turns:
            if e <= ws:
                continue
            if s >= we:
                break
            ov = max(0.0, min(we, e) - max(ws, s))
            if ov > 0:
                overlaps[spk] = overlaps.get(spk, 0.0) + ov
        if not overlaps:
            # Nearest turn by center distance
            wc = (ws + we) / 2
            best_spk = None
            best_d = float("inf")
            for s, e, spk in norm_turns:
                d = min(abs(wc - s), abs(wc - e))
                if d < best_d:
                    best_d = d
                    best_spk = spk
            return [(best_spk, 0.0)] if best_spk else []
        sorted_items = sorted(overlaps.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) >= 2 and sorted_items[1][1] >= boundary_split_min_sec:
            return sorted_items[:2]
        return sorted_items[:1]

    new_segments: list[dict] = []
    current: dict | None = None

    def _flush():
        nonlocal current
        if current is not None and current["words"]:
            current["start"] = current["words"][0].get("start", current.get("start"))
            current["end"] = current["words"][-1].get("end", current.get("end"))
            current["text"] = "".join(w.get("word", "") for w in current["words"]).strip()
            new_segments.append(current)
        current = None

    def _append_word(spk: str, word: dict):
        nonlocal current
        if current is None or current["speaker"] != spk:
            _flush()
            current = {"speaker": spk, "words": [], "start": word.get("start"), "end": word.get("end")}
        current["words"].append(word)

    for seg in result.get("segments", []):
        seg_words = seg.get("words") or []
        if not seg_words:
            # No word-level timestamps (short or unaligned segments).
            # Attribute the whole segment by its own midpoint.
            sstart = seg.get("start")
            send = seg.get("end")
            if sstart is None or send is None:
                continue
            spk_pairs = _word_speaker(float(sstart), float(send))
            if not spk_pairs:
                continue
            spk = spk_pairs[0][0]
            fake_word = {
                "word": seg.get("text", ""),
                "start": sstart,
                "end": send,
            }
            _append_word(spk, fake_word)
            continue

        for w in seg_words:
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                # Word without timestamps (whisperx emits these for
                # punctuation sometimes). Glue it onto the current
                # speaker's run — it has no independent boundary
                # information anyway.
                if current is not None:
                    current["words"].append(w)
                continue
            pairs = _word_speaker(float(ws), float(we))
            if not pairs:
                continue
            if len(pairs) == 1:
                _append_word(pairs[0][0], dict(w))
                continue
            # Split across the boundary: duplicate the word into both
            # speakers' runs. Splitting the glyph itself is unsafe for
            # non-ASCII text, so each half keeps the full word and
            # callers know to dedup with a downstream pass.
            for spk, _ in pairs:
                _append_word(spk, dict(w))

    _flush()

    result["segments"] = new_segments

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to assign speakers from turns: {elapsed_time:.2f} ms")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

def identify_speaker_for_segment(segment_embedding, known_embeddings, threshold=0.1):
    """
    Compare segment_embedding to known speaker embeddings using cosine similarity.
    Returns the speaker name with the highest similarity above the threshold,
    or "Unknown" if none match.
    """
    best_match = "Unknown"
    best_similarity = -1
    for speaker, known_emb in known_embeddings.items():
        similarity = 1 - cosine(segment_embedding, known_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity