# Base image with Python 3.9 and CUDA support for LoFTR acceleration
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Backend packages (api/, src/) live under /app/server
ENV PYTHONPATH=/app/server

# Install system dependencies (OpenCV requires libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (adding missing ones like fastapi, uvicorn)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn reportlab

# Copy the entire project
COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Create a startup script to run both services
RUN echo '#!/bin/bash\n\
uvicorn api.main:app --host 0.0.0.0 --port 8000 &\n\
streamlit run client/app.py --server.port 8501 --server.address 0.0.0.0\n\
wait -n\n\
' > start.sh && chmod +x start.sh

# Start both services
CMD ["./start.sh"]
