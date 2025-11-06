ACRE Offline LLM Switcher (Multi-backend)

## Installation

### macOS (Homebrew)
FIRST: /opt/homebrew/bin/python3 -m pip install --user --break-system-packages -r requirements.txt
THEN: /opt/homebrew/bin/python3 app.py

### Standard Python venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

### NVIDIA Jetson
For NVIDIA Jetson devices, see [JETSON_SETUP.md](JETSON_SETUP.md) for detailed instructions.

**Jetson Orin Nano (JetPack 6.2):**
```bash
chmod +x install_jetson_orin_nano.sh
./install_jetson_orin_nano.sh
```

**Other Jetson models:**
```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

Installed backends:
- GGUF via llama.cpp (default)
- Transformers local
- GPTQ local

Model folder rules:
- file.gguf -> llama.cpp
- folder with quantize_config.json -> AutoGPTQ (NEED TO TEST THIS)
- folder with config.json (+ *.bin or *.safetensors) -> Transformers
- folder with a .gguf inside -> llama.cpp

Run:
python app.py

## Platform Support

- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (x86_64)
- ✅ NVIDIA Jetson (ARM64) - See [JETSON_SETUP.md](JETSON_SETUP.md)

## Notes

- MLX models are Apple Silicon only (not supported on Jetson)
- PyTorch on Jetson must be installed from NVIDIA's wheels (not PyPI)
- GGUF models work best on Jetson devices

TO DO:
containerize
encrypt locally
bundle dependencies -> vendor? wheelhouse? nvidia torch
rename chats / models
rework login
optimize? // tokens
more backend / model types and testing

clean everything up
settings? color scheme? 
