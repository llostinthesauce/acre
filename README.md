# acre

LLM model comparison and workflow testing

## Overview

ACRE (LLM Switchboard) is a flexible platform for comparing and benchmarking different language models. Enables rapid model switching, pipeline toggling, and A/B testing. Supports standard chat, document analysis, and image generation workflows with adjustable parameters and safety features. Designed for offline operation with local model hosting.

## Tech Stack

- Python 3.11+
- Hugging Face Transformers
- GGUF model support
- Flask/FastAPI
- Model training/fine-tuning hooks

## Installation

```bash
# Clone the repository
git clone https://github.com/llostinthesauce/acre.git
cd acre

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Download models** from [Hugging Face](https://huggingface.co/)
   - Recommended starter: [`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
   - Place models in the `models/` directory

3. **Access the interface** - The application will start on localhost

4. **Features:**
   - Swap models on the fly
   - Compare inference quality
   - Adjust tokens/temperature
   - Enable safety checks
   - Run document analysis
   - Toggle different pipelines

> **Note:** Jetson devices use a separate branch with specialized startup scripts.

## License

Personal project - all rights reserved

---

*Part of [@llostinthesauce](https://github.com/llostinthesauce)'s portfolio*
