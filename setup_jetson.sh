#!/bin/bash
# Setup script for NVIDIA Jetson devices
# This script installs PyTorch and other dependencies for Jetson

set -e

echo "=== ACRE Setup for NVIDIA Jetson ==="
echo ""

# Detect Jetson model
if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: This script is for NVIDIA Jetson devices only."
    exit 1
fi

JETSON_MODEL=$(cat /etc/nv_tegra_release | head -n 1 | cut -f2 -d' ' | cut -f1 -d',')
echo "Detected Jetson model: $JETSON_MODEL"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    git \
    wget \
    cmake \
    pkg-config

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson
# PyTorch for Jetson needs to be installed from NVIDIA's wheels
# Check if torch is already installed
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch is already installed:"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo ""
    echo "PyTorch is not installed. Installing PyTorch for Jetson..."
    echo ""
    echo "Please install PyTorch for Jetson using one of these methods:"
    echo ""
    echo "Method 1: Install from NVIDIA's pre-built wheels (recommended)"
    echo "  Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "  Or use:"
    echo "    wget https://nvidia.box.com/shared/static/xxx.whl"
    echo "    pip3 install torch-*.whl"
    echo ""
    echo "Method 2: Install via jetson-containers (if using containers)"
    echo "  See: https://github.com/dusty-nv/jetson-containers"
    echo ""
    echo "Method 3: Build from source (advanced)"
    echo "  See: https://pytorch.org/docs/stable/notes/arm.html"
    echo ""
    read -p "Press Enter after you have installed PyTorch, or Ctrl+C to exit..."
fi

# Verify PyTorch installation
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "PyTorch verification:"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
    fi
else
    echo "ERROR: PyTorch installation verification failed."
    exit 1
fi

# Install base requirements (excluding torch/torchvision)
echo ""
echo "Installing base Python dependencies..."
python3 -m pip install --user \
    customtkinter>=5.2.0 \
    darkdetect>=0.8.0 \
    pillow>=10.2.0 \
    cryptography>=42.0.0 \
    send2trash>=1.8.3 \
    protobuf>=3.20,<6 \
    soundfile>=0.12 \
    pymupdf>=1.24

# Install llama-cpp-python with CUDA support for Jetson
echo ""
echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3 -m pip install --user llama-cpp-python>=0.3.16

# Install transformers and related packages (if torch is available)
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "Installing transformers and related packages..."
    python3 -m pip install --user \
        transformers>=4.52.4 \
        diffusers>=0.25 \
        safetensors>=0.4 \
        huggingface_hub>=0.23 \
        accelerate>=0.31
else
    echo "Skipping transformers installation (PyTorch not available)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now run the application with:"
echo "  python3 app.py"
echo ""
echo "Note: MLX models are not supported on Jetson (Apple Silicon only)."
echo "      Use GGUF, Transformers, or GPTQ models instead."

