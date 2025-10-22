ACRE Offline LLM Switcher (Multi-backend)

can run in a .venv, or locally with homebrew. 
FIRST: /opt/homebrew/bin/python3 -m pip install --user --break-system-packages -r requirements.txt
THEN: /opt/homebrew/bin/python3 app.py

or

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

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
