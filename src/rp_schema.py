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
    'debug': {
        'type': bool,
        'required': False,
        'default': False
    },
    # Chunked-transcription mode: this audio file is a slice of a larger
    # episode starting at this offset (seconds). Every start/end in the
    # output is rebased into the episode's absolute frame before return.
    'audio_offset_sec': {
        'type': float,
        'required': False,
        'default': 0.0
    }
}