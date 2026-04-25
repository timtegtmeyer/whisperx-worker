#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define cache and model directories
CACHE_DIR="/cache/models"
MODELS_DIR="/models"

# Ensure necessary directories exist
mkdir -p /root/.cache/torch/hub/checkpoints

# Download function with caching
download() {
  local file_url="$1"
  local destination_path="$2"
  local cache_path="${CACHE_DIR}/${destination_path##*/}"

  mkdir -p "$(dirname "$cache_path")"
  mkdir -p "$(dirname "$destination_path")"

  if [ ! -e "$cache_path" ]; then
    echo "Downloading $file_url to cache..."
    wget -O "$cache_path" "$file_url"
  else
    echo "Using cached version of ${cache_path##*/}"
  fi

  cp "$cache_path" "$destination_path"
}

# ===============================
# Download Faster Whisper Model
# ===============================
faster_whisper_model_dir="${MODELS_DIR}/faster-whisper-large-v3"
mkdir -p "$faster_whisper_model_dir"

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json"              "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin"              "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json"          "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json"         "$faster_whisper_model_dir/vocabulary.json"

echo "Faster Whisper model downloaded."

# ===================================
# VAD and wav2vec2 are Docker-handled
# ===================================

# Speaker / diarization models (ECAPA, pyannote/embedding,
# pyannote/speaker-diarization-community-1) were removed when ASR was
# decoupled from diarization. The Diarization runs in its own RunPod
# worker; this image is ASR-only.
echo "All models downloaded successfully."