import os

# Import llama_cpp_python
try:
    from llama_cpp import Llama
    llama_cpp_available = True
except ImportError:
    llama_cpp_available = False
    print("⚠️ Llama_cpp not installed.")

# Find and list LLMs based on files found
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
            # Only list files of the appropriate type
            if f.endswith((".gguf", ".bin", ".safetensors")) or os.path.isdir(path):
                files.append(f)
        return files
    
    # Function to load LLMs from files
    def load_model(self, filename):
        if not llama_cpp_available:
            print("⚠️ Llama_cpp not available")
            return None
        
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            print(f"❌ model not found: {path}")
            return None

        print(f"Loading model: {path}")
        try:
            self.current_model = Llama(
                model_path=path,
                # n_gpu_layers=-1, # Uncomment to use GPU acceleration
                # seed=1337, # Uncomment to set a specific seed
                # n_ctx=2048, # Uncomment to increase the context window
            )
            self.backend = "Llama_cpp"
            print("✅ model loaded with Llama_cpp")
            return self.current_model
        except Exception as e:
            print(f"❌ failed to load model: {e}")
            self.current_model = None
            return None

    def generate(self, prompt):
        if not self.current_model:
            return "⚠️ no model loaded yet"
        print(prompt)
        try:
            output = self.current_model(
                prompt, # Prompt
                max_tokens=32, # Max tokens to generate, set to None to generate up to the end of the context window
                # temperature=0.7, # Controls the randomness of the output
                # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                echo=False # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion
            generated_text = output["choices"][0]["text"]
            return(generated_text)
        except Exception as e:
            return f"❌ error during generation: {e}"