FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /

# System packages: Python 3.10, FFmpeg, utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /cache/models /root/.cache/torch

# Copy only requirements file first to leverage Docker cache
COPY builder/requirements.txt /builder/requirements.txt

# Install Python dependencies
RUN python3 -m pip install hf_transfer \
 && python3 -m pip install --no-cache-dir -r /builder/requirements.txt

# Copy the local VAD model to the expected location
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Copy the rest of the builder files
COPY builder /builder

# Download Faster Whisper Models at build time
RUN chmod +x /builder/download_models.sh
RUN /builder/download_models.sh

# Permanently upgrade Lightning checkpoint to silence startup warning
# Upgrade Lightning checkpoint to silence startup warning (path varies by Python version)
RUN python3 -m lightning.pytorch.utilities.upgrade_checkpoint \
    $(python3 -c "import whisperx, os; print(os.path.join(os.path.dirname(whisperx.__file__), 'assets', 'pytorch_model.bin'))") || true

# Copy source code
COPY src .

CMD [ "python3", "-u", "/rp_handler.py" ]
