import os

# import ctransformers if available
try:
    from ctransformers import AutoModelForCausalLM
    ctransformers_available = True
except ImportError:
    ctransformers_available = False
    print("⚠️ ctransformers not installed. models cannot be loaded.")

class model_manager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.current_model = None
        # ensure models folder exists
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def list_models(self):
        """returns a list of .gguf or .bin files in the models folder"""
        files = []
        try:
            for f in os.listdir(self.models_dir):
                if f.endswith(".gguf") or f.endswith(".bin"):
                    files.append(f)
        except FileNotFoundError:
            # folder missing? return empty list
            pass
        return files

    def load_model(self, filename):
        """loads the selected model (only call when user selects one)"""
        if not ctransformers_available:
            print("cannot load model: ctransformers missing")
            return None

        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            print(f"model file not found: {path}")
            return None

        print("loading model:", path)
        # only load when called
        self.current_model = AutoModelForCausalLM.from_pretrained(
            path,
            model_type="llama"
        )
        return self.current_model

    def generate(self, prompt):
        """generate text from the loaded model"""
        if not self.current_model:
            return "no model loaded yet"
        output = ""
        for t in self.current_model(prompt, stream=True):
            output += t
        return output