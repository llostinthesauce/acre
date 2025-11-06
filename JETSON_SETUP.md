# NVIDIA Jetson Setup Guide

This guide will help you set up ACRE on an NVIDIA Jetson device.

## Prerequisites

- NVIDIA Jetson device (Nano, Xavier, AGX, Orin, etc.)
- JetPack installed (Ubuntu-based)
- Python 3.8 or higher
- Internet connection for initial setup

## Quick Setup for Jetson Orin Nano (JetPack 6.2)

**Complete automated installation:**
```bash
chmod +x install_jetson_orin_nano.sh
./install_jetson_orin_nano.sh
```

This script will:
- Install all system dependencies
- Guide you through PyTorch installation
- Install all Python packages
- Set up llama-cpp-python with CUDA support

**For PyTorch installation only:**
```bash
chmod +x install_pytorch_jp62.sh
./install_pytorch_jp62.sh
```

## General Setup (Other Jetson Models)

1. **Run the setup script:**
   ```bash
   chmod +x setup_jetson.sh
   ./setup_jetson.sh
   ```

2. **If PyTorch is not installed, follow one of these methods:**

### Method 1: Install PyTorch from NVIDIA Pre-built Wheels (Recommended)

NVIDIA provides pre-built PyTorch wheels for Jetson. Check the latest version at:
- [NVIDIA Developer Forums - PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

Example installation:
```bash
# Download the wheel file for your JetPack version
wget https://nvidia.box.com/shared/static/[FILENAME].whl
pip3 install torch-[VERSION]-linux_aarch64.whl
```

### Method 2: Install via jetson-containers

If you're using containers:
```bash
# See: https://github.com/dusty-nv/jetson-containers
```

### Method 3: Build PyTorch from Source (Advanced)

See the [PyTorch ARM documentation](https://pytorch.org/docs/stable/notes/arm.html) for building from source.

## Manual Installation Steps

If you prefer to install manually:

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-dev build-essential \
       libopenblas-base libopenmpi-dev libomp-dev git wget cmake pkg-config
   ```

2. **Upgrade pip:**
   ```bash
   python3 -m pip install --upgrade pip setuptools wheel
   ```

3. **Install PyTorch for Jetson** (see methods above)

4. **Install base dependencies:**
   ```bash
   python3 -m pip install --user \
       customtkinter>=5.2.0 \
       darkdetect>=0.8.0 \
       pillow>=10.2.0 \
       cryptography>=42.0.0 \
       send2trash>=1.8.3 \
       protobuf>=3.20,<6 \
       soundfile>=0.12 \
       pymupdf>=1.24
   ```

5. **Install llama-cpp-python with CUDA support:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3 -m pip install --user llama-cpp-python>=0.3.16
   ```

6. **Install transformers and related packages:**
   ```bash
   python3 -m pip install --user \
       transformers>=4.52.4 \
       diffusers>=0.25 \
       safetensors>=0.4 \
       huggingface_hub>=0.23 \
       accelerate>=0.31
   ```

## Verify Installation

Check that PyTorch is working with CUDA:
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the Application

```bash
python3 app.py
```

## Important Notes

- **MLX models are NOT supported** on Jetson (MLX is Apple Silicon only)
- Use **GGUF models** (via llama.cpp) for best performance on Jetson
- **Transformers models** work but may be slower
- **GPTQ models** may work but require additional setup
- Make sure to set device preference to **"cuda"** in settings for GPU acceleration

## Troubleshooting

### PyTorch not found
- Ensure PyTorch is installed for Jetson (not the standard PyPI version)
- Verify with: `python3 -c "import torch; print(torch.__version__)"`

### CUDA not available
- Check that JetPack is properly installed
- Verify CUDA: `nvcc --version`
- Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### llama-cpp-python build fails
- Ensure you have build tools: `sudo apt-get install build-essential cmake`
- Try building with CUDA: `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`

### Out of memory errors
- Jetson devices have limited VRAM
- Use smaller models or quantized models (GGUF Q4, Q5, Q8)
- Reduce max_tokens in settings
- Consider using CPU for very large models

## Performance Tips

1. **Use GGUF models** - They're optimized for edge devices
2. **Use quantized models** - Q4, Q5, or Q8 quantization
3. **Set device to CUDA** - Enable GPU acceleration in settings
4. **Monitor memory** - Use `tegrastats` to monitor GPU/CPU usage
5. **Power mode** - Set Jetson to MAXN mode for best performance:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

## Supported Backends on Jetson

- ✅ **llama.cpp (GGUF)** - Recommended, best performance
- ✅ **Transformers** - Works with CUDA acceleration
- ⚠️ **AutoGPTQ** - May require additional setup
- ✅ **Diffusers** - For image generation (if you have enough VRAM)
- ❌ **MLX** - Not supported (Apple Silicon only)

## Getting Help

If you encounter issues:
1. Check the [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
2. Verify your JetPack version matches PyTorch requirements
3. Check that all system dependencies are installed

