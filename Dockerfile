FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py model.py main.py ./  

# Default command to run training
ARG MODE=train
ARG MODE=train
ENV MODE=${MODE}
CMD ["sh", "-c", "python $MODE.py"]