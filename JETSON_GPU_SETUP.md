# Jetson GPU Configuration Guide

## Quick Fix: Enable GPU for llama.cpp Models

The app should now automatically detect Jetson and use GPU, but if you're still seeing "assigned to device CPU" in the logs, follow these steps:

### 1. Verify CUDA is Available

```bash
# Check if nvidia-smi works
nvidia-smi

# Check if CUDA libraries are available
ls /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null || echo "CUDA not found in standard location"
```

### 2. Verify llama-cpp-python was Built with CUDA

```bash
python3 -c "from llama_cpp import Llama; print('llama_cpp imported successfully')"
```

If you see errors about CUDA, you need to reinstall llama-cpp-python with CUDA support:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install --user --force-reinstall llama-cpp-python
```

### 3. Set Device Preference in App

1. Open the app
2. Go to **Settings** tab
3. Set **Device preference** to **"cuda"** (not "auto")
4. Save settings
5. Reload your model

### 4. Verify GPU Usage

After loading a model, check the terminal output. You should see:
```
load_tensors: layer   0 assigned to device CUDA, is_swa = 0
load_tensors: layer   1 assigned to device CUDA, is_swa = 0
...
```

If you still see "CPU", the model is running on CPU.

## Troubleshooting

### All layers still on CPU

1. **Check device preference**: Make sure it's set to "cuda" in settings
2. **Check llama-cpp-python build**: Reinstall with CUDA support (see above)
3. **Check CUDA availability**: Run `nvidia-smi` to verify GPU is accessible
4. **Restart the app**: After changing settings, restart the application

### Performance Issues

- **Use quantized models**: Q4_K_M, Q5_K_M, or Q8_0 quantizations work best
- **Monitor GPU usage**: Run `tegrastats` in another terminal to see GPU utilization
- **Set power mode**: For best performance:
  ```bash
  sudo nvpmodel -m 0  # MAXN mode
  sudo jetson_clocks  # Max clocks
  ```

### Model Loading Errors

If you get errors about GPU memory:
- Use a smaller model or more aggressive quantization
- Reduce context size in settings
- Some models may be too large for Jetson's VRAM

## Expected Behavior

When GPU is working correctly:
- Model layers load to CUDA device
- Generation is faster than CPU
- `tegrastats` shows GPU activity during generation
- Lower CPU usage during inference

## Manual Verification

Test GPU detection:
```bash
python3 -c "
from model_manager.manager import ModelManager
mgr = ModelManager(device_pref='cuda')
print(f'GPU layers: {mgr._llama_gpu_layers}')
"
```

Should print: `GPU layers: -1` (meaning all layers on GPU)

