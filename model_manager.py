import os

try:
    from ctransformers import AutoModelForCausalLM
    ctransformers_available = True
except ImportError:
    ctransformers_available = False
    print("⚠️ ctransformers not installed.")

class model_manager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.current_model = None
        self.backend = None

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def list_models(self):
        files = []
        for f in os.listdir(self.models_dir):
            path = os.path.join(self.models_dir, f)
            if f.endswith((".gguf", ".bin", ".safetensors")) or os.path.isdir(path):
                files.append(f)
        return files

    def load_model(self, filename):
        if not ctransformers_available:
            print("⚠️ ctransformers not available")
            return None

        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            print(f"❌ model not found: {path}")
            return None

        print(f"Loading model: {path}")
        try:
            self.current_model = AutoModelForCausalLM.from_pretrained(
                path,
                model_type="llama",
                local_files_only=True
            )
            self.backend = "ctransformers"
            print("✅ model loaded with ctransformers")
            return self.current_model
        except Exception as e:
            print(f"❌ failed to load model: {e}")
            self.current_model = None
            return None

    def generate(self, prompt, max_tokens=256):
        if not self.current_model:
            return "⚠️ no model loaded yet"

        try:
            output = ""
            for token in self.current_model(prompt, stream=True, max_new_tokens=max_tokens):
                output += token
            return output
        except Exception as e:
            return f"❌ error during generation: {e}"