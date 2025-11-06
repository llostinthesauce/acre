#!/bin/bash
# Quick PyTorch installation script for JetPack 6.2
# This script installs PyTorch using the jetson-containers method (recommended)

set -e

echo "=========================================="
echo "PyTorch Installation for JetPack 6.2"
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

# Method 1: Try jetson-containers (recommended)
echo "Method 1: Installing via jetson-containers (Recommended)"
echo "This is the most reliable method for JetPack 6.2"
echo ""
read -p "Install jetson-containers and PyTorch? (Y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Installing jetson-containers..."
    curl -sL https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install.sh | bash
    
    echo ""
    echo "jetson-containers installed. To use PyTorch:"
    echo "  ./run.sh --name pytorch"
    echo ""
    echo "Or install PyTorch directly in your current environment:"
    echo "  ./run.sh --name pytorch --install-packages"
    echo ""
fi

# Method 2: Manual wheel installation
echo ""
echo "Method 2: Manual wheel installation"
echo "If jetson-containers doesn't work, you can download wheels manually:"
echo ""
echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "2. Find the wheel for:"
echo "   - JetPack 6.2"
echo "   - Python $PYTHON_MAJOR_MINOR"
echo "   - Architecture: aarch64"
echo ""
echo "3. Download and install:"
echo "   wget [WHEEL_URL]"
echo "   pip3 install --user [WHEEL_FILE]"
echo ""

# Check if PyTorch is now available
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is installed!"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            python3 -c "import torch; print(f'  CUDA device: {torch.cuda.get_device_name(0)}')"
        fi
    fi
else
    echo "⚠ PyTorch is not yet installed."
    echo "  Please use one of the methods above."
fi

echo ""
echo "=========================================="

