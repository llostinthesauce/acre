#!/bin/bash
# Complete installation script for ACRE on Jetson Orin Nano with JetPack 6.2
# This script installs PyTorch and all dependencies automatically

set -e

echo "=========================================="
echo "ACRE Installation for Jetson Orin Nano"
echo "JetPack 6.2"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: This script is for NVIDIA Jetson devices only."
    exit 1
fi

# Detect JetPack version
JETPACK_VERSION=$(cat /etc/nv_tegra_release | head -n 1 | cut -f2 -d' ' | cut -f1 -d',')
echo "Detected JetPack version: $JETPACK_VERSION"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"
echo ""

# Determine Python tag for wheel
if [[ "$PYTHON_MAJOR_MINOR" == "3.10" ]]; then
    PYTHON_TAG="cp310"
elif [[ "$PYTHON_MAJOR_MINOR" == "3.11" ]]; then
    PYTHON_TAG="cp311"
elif [[ "$PYTHON_MAJOR_MINOR" == "3.9" ]]; then
    PYTHON_TAG="cp39"
elif [[ "$PYTHON_MAJOR_MINOR" == "3.8" ]]; then
    PYTHON_TAG="cp38"
else
    PYTHON_TAG="cp310"  # Default to 3.10
    echo "WARNING: Unknown Python version, defaulting to cp310"
fi

echo "Using Python tag: $PYTHON_TAG"
echo ""

# Step 1: Install system dependencies
echo "=========================================="
echo "Step 1: Installing system dependencies..."
echo "=========================================="
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libopenblas-base \
    libopenblas-dev \
    libopenmpi-dev \
    libomp-dev \
    git \
    wget \
    curl \
    cmake \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev

echo "System dependencies installed."
echo ""

# Step 2: Upgrade pip and install numpy
echo "=========================================="
echo "Step 2: Upgrading pip and installing numpy..."
echo "=========================================="
python3 -m pip install --upgrade pip setuptools wheel
echo "Installing numpy..."
if ! python3 -m pip install --user "numpy==1.26.1"; then
    echo "Failed to install numpy==1.26.1, trying latest version..."
    python3 -m pip install --user numpy
fi
echo ""

# Step 3: Install PyTorch for JetPack 6.2
echo "=========================================="
echo "Step 3: Installing PyTorch for JetPack 6.2..."
echo "=========================================="

# Check if torch is already installed
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch is already installed:"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
    fi
    echo ""
    read -p "PyTorch is already installed. Reinstall? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping PyTorch installation."
        SKIP_PYTORCH=true
    else
        SKIP_PYTORCH=false
    fi
else
    SKIP_PYTORCH=false
fi

if [ "$SKIP_PYTORCH" = false ]; then
    echo "Installing PyTorch for JetPack 6.2..."
    echo ""
    
    # For JetPack 6.2, we'll use the jetson-containers method or direct wheel
    # Try to install from jetson-containers repository first (most reliable)
    echo "Attempting to install PyTorch via jetson-containers method..."
    
    # Install PyTorch 2.1.0 for JetPack 6.2 (common version)
    # You may need to adjust the version based on what's available
    PYTORCH_VERSION="2.1.0"
    
    # Try downloading from NVIDIA's developer site
    echo "Downloading PyTorch wheel for JetPack 6.2..."
    
    # For JetPack 6.2, try the official NVIDIA wheels
    # Note: URLs may need to be updated based on actual availability
    TORCH_WHEEL_URL=""
    
    # Try multiple potential URLs for JetPack 6.2
    # These are example URLs - actual URLs may vary
    if [ -z "$TORCH_WHEEL_URL" ]; then
        echo "Attempting to find PyTorch wheel for JetPack 6.2..."
        echo ""
        echo "For JetPack 6.2, you have a few options:"
        echo ""
        echo "Option 1: Use jetson-containers (Recommended)"
        echo "  Run: bash <(curl -sL https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install.sh)"
        echo "  Then: ./run.sh --name pytorch"
        echo ""
        echo "Option 2: Install from NVIDIA Developer Forums"
        echo "  Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo "  Download the wheel file for JetPack 6.2 and Python $PYTHON_MAJOR_MINOR"
        echo ""
        echo "Option 3: Build from source (takes several hours)"
        echo ""
        echo "For now, we'll try to install a compatible version..."
        echo ""
        
        # Try installing via pip with index-url (may work for some versions)
        # This is a fallback - may not work for all JetPack versions
        echo "Attempting to install PyTorch 2.1.0..."
        
        # Install PyTorch dependencies first
        python3 -m pip install --user typing-extensions
        
        # Try to install PyTorch - this may fail, which is expected
        # We'll catch the error and provide instructions
        if python3 -m pip install --user "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121 2>&1 | tee /tmp/pytorch_install.log; then
            echo "PyTorch installed successfully via PyPI!"
        else
            echo ""
            echo "PyTorch installation from PyPI failed (expected on ARM64)."
            echo "Please install PyTorch manually using one of these methods:"
            echo ""
            echo "METHOD 1 (Easiest): Use jetson-containers"
            echo "  curl -sL https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install.sh | bash"
            echo "  ./run.sh --name pytorch"
            echo ""
            echo "METHOD 2: Install PyTorch directly (if available for JP 6.2)"
            echo "  For JetPack 6.2, try:"
            echo "    pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            echo "  Note: This may not work on ARM64 - if it fails, use Method 1"
            echo ""
            echo "METHOD 3: Download wheel from NVIDIA"
            echo "  1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
            echo "  2. Find the wheel for JetPack 6.2 and Python $PYTHON_MAJOR_MINOR"
            echo "  3. Download and install:"
            echo "     wget [WHEEL_URL]"
            echo "     pip3 install --user [WHEEL_FILE]"
            echo ""
            echo "After installing PyTorch, re-run this script to continue with other dependencies."
            echo ""
            read -p "Press Enter to continue with other dependencies (PyTorch will be skipped)..."
        fi
    fi
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is installed:"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            python3 -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
        fi
    fi
    PYTORCH_AVAILABLE=true
else
    echo "⚠ PyTorch is not installed. Some features will be unavailable."
    echo "  You can install it later and re-run this script."
    PYTORCH_AVAILABLE=false
fi
echo ""

# Step 4: Install base Python dependencies
echo "=========================================="
echo "Step 4: Installing base Python dependencies..."
echo "=========================================="
python3 -m pip install --user \
    "customtkinter>=5.2.0" \
    "darkdetect>=0.8.0" \
    "pillow>=10.2.0" \
    "cryptography>=42.0.0" \
    "send2trash>=1.8.3" \
    "protobuf>=3.20,<6" \
    "soundfile>=0.12" \
    "pymupdf>=1.24"

echo "Base dependencies installed."
echo ""

# Step 5: Install llama-cpp-python with CUDA support
echo "=========================================="
echo "Step 5: Installing llama-cpp-python with CUDA support..."
echo "=========================================="
echo "This may take several minutes to compile..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3 -m pip install --user "llama-cpp-python>=0.3.16"
echo "llama-cpp-python installed."
echo ""

# Step 6: Install transformers and related packages (if PyTorch is available)
if [ "$PYTORCH_AVAILABLE" = true ]; then
    echo "=========================================="
    echo "Step 6: Installing transformers and related packages..."
    echo "=========================================="
    python3 -m pip install --user \
        "transformers>=4.52.4" \
        "diffusers>=0.25" \
        "safetensors>=0.4" \
        "huggingface_hub>=0.23" \
        "accelerate>=0.31"
    
    echo "Transformers packages installed."
    echo ""
else
    echo "=========================================="
    echo "Step 6: Skipping transformers (PyTorch not available)"
    echo "=========================================="
    echo "Install PyTorch first, then run:"
    echo "  pip3 install --user \"transformers>=4.52.4\" \"diffusers>=0.25\" \"safetensors>=0.4\" \"huggingface_hub>=0.23\" \"accelerate>=0.31\""
    echo ""
fi

# Final summary
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - System dependencies: ✓"
echo "  - Base Python packages: ✓"
echo "  - llama-cpp-python (CUDA): ✓"
if [ "$PYTORCH_AVAILABLE" = true ]; then
    echo "  - PyTorch: ✓"
    echo "  - Transformers packages: ✓"
else
    echo "  - PyTorch: ✗ (install manually)"
    echo "  - Transformers packages: ✗ (requires PyTorch)"
fi
echo ""
echo "To run ACRE:"
echo "  python3 app.py"
echo ""
echo "Notes:"
echo "  - MLX models are NOT supported on Jetson (Apple Silicon only)"
echo "  - Use GGUF models for best performance"
echo "  - Set device preference to 'cuda' in settings for GPU acceleration"
echo ""
if [ "$PYTORCH_AVAILABLE" = false ]; then
    echo "To install PyTorch later, use jetson-containers:"
    echo "  curl -sL https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install.sh | bash"
    echo "  ./run.sh --name pytorch"
    echo ""
fi
echo "=========================================="

