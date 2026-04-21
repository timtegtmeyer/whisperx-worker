INPUT_VALIDATIONS = {
    'audio_file': {
        'type': str,
        'required': True
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'language_detection_min_prob': {
        'type': float,
        'required': False,
        'default': 0
    },
    'language_detection_max_tries': {
        'type': int,
        'required': False,
        'default': 5
    },
    'initial_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 16
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0
    },
    'vad_onset': {
        'type': float,
        'required': False,
        'default': 0.500
    },
    'vad_offset': {
        'type': float,
        'required': False,
        'default': 0.363
    },
    'align_output': {
        'type': bool,
        'required': False,
        'default': False
    },
    'diarization': {
        'type': bool,
        'required': False,
        'default': False
    },
    'huggingface_access_token': {
        'type': str,
        'required': False,
        'default': None
    },
    'min_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'debug': {
        'type': bool,
        'required': False,
        'default': False
    },
    'speaker_verification': {
        'type': bool,
        'required': False,
        'default': False
    },
    'speaker_samples': {
        'type': list,
        'required': False,
        'default': []
    },
    'speaker_match_threshold': {
        'type': float,
        'required': False,
        'default': 0.85
    },
    # Chunked-transcription mode: this audio file is a slice of a larger
    # episode starting at this offset (seconds). Every start/end in the
    # output is rebased into the episode's absolute frame before return,
    # so the caller can stitch N chunk outputs together with monotonic
    # timestamps. Default 0 = treat the file as the full episode (back-
    # compat with all existing callers).
    'audio_offset_sec': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    # Chunked-transcription mode: pre-computed pyannote diarization turns
    # for this chunk's portion of the audio. Shape:
    #   [{"start": float, "end": float, "speaker": str}, ...]
    # When provided, the worker SKIPS its own DiarizationPipeline model
    # load + inference and uses these turns directly to call
    # whisperx.assign_word_speakers — big VRAM + time saving, and lets
    # chunked jobs share episode-stable speaker IDs assigned by pyannote
    # running once over the full audio up front.
    'diarization_turns': {
        'type': list,
        'required': False,
        'default': None
    }
}