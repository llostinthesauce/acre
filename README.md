# ACRE is a switchboard for running and comparing LLM workflows. It lets you swap models on the fly (including imports), toggle pipelines, and run standard chat/inference, document analysis, and image generation. You can adjust tokens/temperature, pick themes, and enable safety checks like query parsing. Model training hooks are available when you want to fine-tune. Note: Jetson devices use a different startup script and live on their own Jetson branch.

# ACRE LLM Switchboard

## How to Use

1. **Clone the repository** from GitHub to your device.

2. **Ensure that Python is installed.**  
   Python **3.11+** is required to run the application.

3. **Open a terminal** and navigate to the location where you cloned the repository, for example:  
   ```bash
   cd path/to/acre
   ```

4. *(Optional but recommended)* **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   On Windows (PowerShell), you might use:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

5. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This command will install all of the required dependencies to run the app, which are listed in the `requirements.txt` file.

6. You can now run the application in **offline mode**.  
   Depending on your preferences, you may disconnect from your network or stay connected.  
   However, to **download models**, complete step 7 **before** disconnecting your device from the network.

7. **Run the application:**
   ```bash
   python app.py
   ```
   If that does not work, try:
   ```bash
   py app.py
   ```
   or:
   ```bash
   python3 app.py
   ```

8. **Download models** from [Hugging Face](https://huggingface.co/) or by clicking the link located within the Settings menu in the application.  
   Recommended for testing:
   - [`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)

9. **Add/load models.**  
   For easy access, place downloaded model files in the `models/` folder located within the application directory.
