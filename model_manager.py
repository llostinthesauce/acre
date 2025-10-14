import json
import os
import re
from glob import glob

# Optional backends
try:
    from llama_cpp import Llama
    llama_cpp_available = True
except ImportError:
    Llama = None
    llama_cpp_available = False
    print("⚠️ llama_cpp not installed.")

try:
    from ctransformers import AutoModelForCausalLM
    ctransformers_available = True
except ImportError:
    AutoModelForCausalLM = None
    ctransformers_available = False
    print("⚠️ ctransformers not installed.")


class model_manager:
    def __init__(self, models_dir="models", history_dir="history"):
        self.models_dir = models_dir
        self.history_dir = history_dir
        self.current_model = None
        self.current_model_name = None
        self.backend = None
        self.history_file = None
        self.history = []

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.registry = self._load_registry()

    def list_models(self):
        files = []
        for f in os.listdir(self.models_dir):
            path = os.path.join(self.models_dir, f)
            if f.endswith((".gguf", ".bin", ".safetensors")) or os.path.isdir(path):
                files.append(f)
        return files

    def load_model(self, filename):
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            print(f"❌ model not found: {path}")
            return None

        registry_entry = self._lookup_registry(filename)
        preferred_backend = None
        if registry_entry:
            preferred_backend = registry_entry.get("backend")

        backends_to_try = self._build_backend_priority(preferred_backend, path)
        if not backends_to_try:
            print("❌ no compatible backend available (install llama_cpp or ctransformers).")
            return None

        load_errors = []
        for backend_name in backends_to_try:
            loader = getattr(self, f"_load_with_{backend_name}", None)
            if not loader:
                continue
            try:
                model = loader(path, registry_entry or {})
                self.current_model = model
                self.backend = backend_name
                self.current_model_name = filename
                self._prepare_history(filename)
                print(f"✅ model loaded with {backend_name}")
                return self.current_model
            except Exception as exc:
                load_errors.append(f"{backend_name}: {exc}")

        self.current_model = None
        self.backend = None
        self.current_model_name = None
        if load_errors:
            joined = "; ".join(load_errors)
            print(f"❌ failed to load model. Tried {len(load_errors)} backend(s). Details: {joined}")
        return None

    def is_loaded(self):
        return self.current_model is not None

    def generate(self, user_prompt, max_tokens=256):
        if not self.current_model:
            return "⚠️ no model loaded yet"

        history_snapshot = list(self.history)
        self.history.append({"role": "user", "content": user_prompt})
        prompt_text = self._build_prompt()
        try:
            if self.backend == "llama_cpp":
                output = self.current_model(
                    prompt_text,
                    max_tokens=max_tokens,
                    echo=False,
                )
                generated_text = output["choices"][0]["text"]
            elif self.backend == "ctransformers":
                generated_text = self.current_model(
                    prompt_text,
                    max_new_tokens=max_tokens,
                )
            else:
                raise RuntimeError(f"Unsupported backend: {self.backend}")

            assistant_response = generated_text.strip()
            self.history.append({"role": "assistant", "content": assistant_response})
            self._save_history()
            return assistant_response
        except Exception as exc:
            self.history = history_snapshot
            print(f"❌ error during generation: {exc}")
            return f"❌ error during generation: {exc}"

    def get_history(self):
        return list(self.history)

    def clear_history(self):
        self.history = []
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except OSError:
                pass

    # Internal helpers -------------------------------------------------
    def _load_registry(self):
        registry_path = os.path.join(self.models_dir, "registry.json")
        if not os.path.exists(registry_path):
            return {}
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            print("⚠️ registry.json is not a JSON object; ignoring.")
        except Exception as exc:
            print(f"⚠️ failed to read registry.json: {exc}")
        return {}

    def _lookup_registry(self, filename):
        if filename in self.registry:
            return self.registry[filename]
        basename = os.path.basename(filename)
        return self.registry.get(basename)

    def _build_backend_priority(self, preferred_backend, path):
        candidates = []
        if preferred_backend == "llama_cpp" and llama_cpp_available:
            candidates.append("llama_cpp")
        elif preferred_backend == "ctransformers" and ctransformers_available:
            candidates.append("ctransformers")

        if not candidates:
            if os.path.isdir(path):
                if ctransformers_available:
                    candidates.append("ctransformers")
            else:
                if path.lower().endswith(".gguf") and llama_cpp_available:
                    candidates.append("llama_cpp")
                if ctransformers_available:
                    candidates.append("ctransformers")
                if llama_cpp_available and "llama_cpp" not in candidates:
                    candidates.append("llama_cpp")

        if not candidates:
            if llama_cpp_available:
                candidates.append("llama_cpp")
            if ctransformers_available:
                candidates.append("ctransformers")

        # Ensure uniqueness while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    def _prepare_history(self, filename):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
        self.history_file = os.path.join(self.history_dir, f"{safe_name}.json")
        self.history = self._load_history_file()

    def _load_history_file(self):
        if not self.history_file or not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception as exc:
            print(f"⚠️ failed to load history {self.history_file}: {exc}")
        return []

    def _save_history(self):
        if not self.history_file:
            return
        try:
            with open(self.history_file, "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"⚠️ failed to save history: {exc}")

    def _build_prompt(self):
        parts = []
        for message in self.history:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _load_with_llama_cpp(self, path, registry_entry):
        if not llama_cpp_available:
            raise RuntimeError("llama_cpp backend unavailable.")

        llama_params = dict(registry_entry.get("llama_cpp_params", {}))
        model_path = path
        if os.path.isdir(path):
            gguf_files = sorted(glob(os.path.join(path, "*.gguf")))
            if not gguf_files:
                raise FileNotFoundError("no .gguf file found inside directory for llama_cpp.")
            model_path = gguf_files[0]

        llama_params.setdefault("model_path", model_path)
        if "model_path" not in llama_params:
            llama_params["model_path"] = model_path

        return Llama(**llama_params)

    def _load_with_ctransformers(self, path, registry_entry):
        if not ctransformers_available:
            raise RuntimeError("ctransformers backend unavailable.")

        params = dict(registry_entry.get("ctransformers_params", {}))
        model_type = registry_entry.get("model_type") or self._infer_model_type(path)
        if model_type:
            params.setdefault("model_type", model_type)

        return AutoModelForCausalLM.from_pretrained(path, **params)

    def _infer_model_type(self, path):
        name = os.path.basename(path).lower()
        if "mistral" in name:
            return "mistral"
        if "falcon" in name:
            return "falcon"
        if "gptq" in name:
            return "gptj"
        if "phi" in name:
            return "phi"
        return "llama"
