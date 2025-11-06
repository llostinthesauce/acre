#!/bin/bash
# PyTorch installation script for Jetson Orin Nano (JetPack 6.2)
# This script uses jetson-containers, the most reliable method

set -e

echo "=========================================="
echo "PyTorch Installation for Jetson"
echo "JetPack 6.2 - Orin Nano"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: This script is for NVIDIA Jetson devices only."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"
echo ""

# Method 1: jetson-containers (Recommended)
echo "=========================================="
echo "Method 1: Installing via jetson-containers"
echo "=========================================="
echo "This is the most reliable method for JetPack 6.2"
echo ""

# Check if jetson-containers is already installed
if [ -d "$HOME/jetson-containers" ]; then
    echo "jetson-containers directory found."
    read -p "Reinstall jetson-containers? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing jetson-containers..."
        rm -rf "$HOME/jetson-containers"
    else
        echo "Using existing jetson-containers installation."
        USE_EXISTING=true
    fi
else
    USE_EXISTING=false
fi

if [ "$USE_EXISTING" != true ]; then
    echo "Installing jetson-containers..."
    curl -sL https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install.sh | bash
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: jetson-containers installation failed."
        echo "Trying alternative method..."
        USE_CONTAINERS=false
    else
        USE_CONTAINERS=true
    fi
else
    USE_CONTAINERS=true
fi

if [ "$USE_CONTAINERS" = true ]; then
    echo ""
    echo "jetson-containers installed successfully!"
    echo ""
    echo "Now installing PyTorch..."
    echo ""
    
    # Navigate to jetson-containers directory
    cd "$HOME/jetson-containers" || exit 1
    
    # Install PyTorch container
    echo "Building/installing PyTorch container..."
    ./run.sh --name pytorch --install-packages
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "PyTorch installed successfully!"
        echo "=========================================="
        echo ""
        echo "To use PyTorch in your current environment, run:"
        echo "  cd ~/jetson-containers"
        echo "  ./run.sh --name pytorch"
        echo ""
        echo "Or to install PyTorch packages in your current Python environment:"
        echo "  cd ~/jetson-containers"
        echo "  ./run.sh --name pytorch --install-packages"
        echo ""
    else
        echo ""
        echo "WARNING: PyTorch installation via containers had issues."
        echo "Trying alternative method..."
        USE_CONTAINERS=false
    fi
fi

# Method 2: Direct wheel installation (fallback)
if [ "$USE_CONTAINERS" != true ]; then
    echo ""
    echo "=========================================="
    echo "Method 2: Direct wheel installation"
    echo "=========================================="
    echo ""
    echo "For JetPack 6.2, you need to download the PyTorch wheel manually."
    echo ""
    echo "Steps:"
    echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "2. Find the wheel for:"
    echo "   - JetPack 6.2"
    echo "   - Python $PYTHON_MAJOR_MINOR"
    echo "   - Architecture: aarch64"
    echo ""
    echo "3. Download the wheel file"
    echo "4. Install it:"
    echo "   pip3 install --user [WHEEL_FILE]"
    echo ""
    echo "Or try the PyTorch index (may not work on ARM64):"
    echo "  pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying PyTorch installation..."
echo "=========================================="

if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is installed!"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            python3 -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
            echo ""
            echo "✓ CUDA is working! PyTorch can use your GPU."
        else
            echo ""
            echo "⚠ CUDA is not available. PyTorch will use CPU only."
        fi
    fi
else
    echo "⚠ PyTorch is not installed yet."
    echo "  Follow the instructions above to complete installation."
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "After PyTorch is installed, install transformers packages:"
echo "  pip3 install --user transformers>=4.52.4 safetensors>=0.4 accelerate>=0.31"
echo ""
echo "Then restart your ACRE application."
echo ""

