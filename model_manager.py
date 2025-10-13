import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    from ctransformers import AutoModelForCausalLM

    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False
    print("⚠️ ctransformers not installed. Install it to load local GGUF models.")

try:
    import torch
    from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None  # type: ignore
    print("ℹ️ transformers/torch not installed. Safetensors backends will be unavailable.")
ALLOWED_EXTENSIONS = (".gguf", ".bin", ".safetensors")
REGISTRY_FILENAME = "registry.json"

# Mapping of friendly architecture names to ctransformers model_type values.
MODEL_TYPE_ALIASES = {
    "llama": "llama",
    "llama3": "llama",
    "llama-3": "llama",
    "llama3.1": "llama",
    "chatgpt-oss": "qwen2",
    "qwen": "qwen",
    "qwen2": "qwen2",
    "deepseek": "deepseek",
    "deepseek-coder": "starcoder",
    "deepseek-coder-v2": "starcoder",
    "mistral": "mistral",
    "mixtral": "mistral",
    "phi": "phi",
    "phi-2": "phi",
    "starcoder": "starcoder",
}


class model_manager:
    """
    Lightweight registry-driven loader for offline LLM weights.
    - All models live inside `models/`.
    - Metadata is stored in `models/registry.json`.
    - Supports local execution via ctransformers (GGUF/bin) or transformers (safetensors).
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.current_model = None
        self.current_tokenizer: Optional[Any] = None
        self.current_model_config: Optional[Dict] = None
        self.backend = None
        self.device: Optional["torch.device"] = None
        self._lock = threading.Lock()

        os.makedirs(self.models_dir, exist_ok=True)
        self.registry_path = os.path.join(self.models_dir, REGISTRY_FILENAME)
        self.registry = self._load_registry()

    # ------------------------------------------------------------------ Registry
    def _load_registry(self) -> Dict:
        if not os.path.exists(self.registry_path):
            registry = {"models": []}
            self._sync_registry_with_filesystem(registry)
            self._save_registry(registry)
            return registry

        try:
            with open(self.registry_path, "r", encoding="utf-8") as fh:
                registry = json.load(fh)
        except (json.JSONDecodeError, OSError):
            registry = {"models": []}

        if "models" not in registry:
            registry["models"] = []

        changed = self._sync_registry_with_filesystem(registry)
        if changed:
            self._save_registry(registry)
        return registry

    def _sync_registry_with_filesystem(self, registry: Dict) -> bool:
        """
        Ensure any loose model files in the models directory are tracked with
        a default configuration so they appear in the GUI.
        """
        tracked_paths = {model["path"] for model in registry.get("models", [])}
        updated = False

        for entry in sorted(os.listdir(self.models_dir)):
            if entry == REGISTRY_FILENAME:
                continue
            full_path = os.path.join(self.models_dir, entry)

            if os.path.isdir(full_path):
                # Directory-based model (already packaged). Require manual registration.
                continue

            if not entry.lower().endswith(ALLOWED_EXTENSIONS):
                continue

            if entry not in tracked_paths:
                registry.setdefault("models", []).append(
                    {
                        "name": os.path.splitext(entry)[0],
                        "path": entry,
                        "backend": "ctransformers",
                        "model_type": self._guess_model_type(entry),
                    }
                )
                updated = True

        return updated

    def _save_registry(self, registry: Optional[Dict] = None) -> None:
        payload = registry or self.registry
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------ Helpers
    def _guess_model_type(self, filename: str) -> str:
        lower = filename.lower()
        for key, model_type in MODEL_TYPE_ALIASES.items():
            if key in lower:
                return model_type
        return "llama"

    def _resolve_model_type(self, value: str) -> str:
        if not value:
            return "llama"
        key = value.lower()
        return MODEL_TYPE_ALIASES.get(key, value)

    def _find_model(self, name: str) -> Optional[Dict]:
        for model in self.registry.get("models", []):
            if model.get("name") == name:
                return model
        return None

    # ------------------------------------------------------------------ Public API
    def refresh(self) -> None:
        """Reload registry from disk so GUI reflects manual edits."""
        self.registry = self._load_registry()

    def list_models(self) -> List[str]:
        self.refresh()
        return [model["name"] for model in self.registry.get("models", [])]

    def describe_model(self, name: str) -> Optional[Dict]:
        return self._find_model(name)

    def register_model(
        self,
        name: str,
        path: str,
        model_type: str,
        backend: str = "ctransformers",
        prompt_template: Optional[str] = None,
    ) -> None:
        self.refresh()
        config = self._find_model(name)
        entry = {
            "name": name,
            "path": path,
            "backend": backend,
            "model_type": self._resolve_model_type(model_type),
        }
        if prompt_template:
            entry["prompt_template"] = prompt_template

        if config:
            config.update(entry)
        else:
            self.registry.setdefault("models", []).append(entry)

        self._save_registry()

    def delete_model(self, name: str) -> bool:
        self.refresh()
        before = len(self.registry.get("models", []))
        self.registry["models"] = [
            model for model in self.registry.get("models", []) if model.get("name") != name
        ]
        removed = len(self.registry["models"]) != before
        if removed:
            self._save_registry()
        return removed

    def load_model(self, name: str):
        config = self._find_model(name)
        if not config:
            print(f"❌ model '{name}' is not registered")
            return None

        backend = config.get("backend", "ctransformers")
        path = config.get("path")
        if not path:
            print(f"❌ no path configured for model '{name}'")
            return None

        resolved_path = os.path.join(self.models_dir, path)
        if not os.path.exists(resolved_path):
            print(f"❌ model weights missing at {resolved_path}")
            return None

        # Reset handles
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_config = None
        self.backend = None
        self.device = None

        model_type = self._resolve_model_type(config.get("model_type", "llama"))

        if backend == "ctransformers":
            if not CTRANSFORMERS_AVAILABLE:
                print("⚠️ ctransformers not available")
                return None

            model_dir, model_file = self._interpret_model_path(resolved_path)
            print(f"Loading model '{name}' from {resolved_path}")
            try:
                load_kwargs = {
                    "model_type": model_type,
                    "local_files_only": True,
                }
                if model_file:
                    load_kwargs["model_file"] = model_file

                loaded_model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
                self.current_model = loaded_model
                self.current_model_config = config
                self.backend = backend
                print(f"✅ model '{name}' ready (backend: {backend}, type: {model_type})")
                return loaded_model
            except Exception as exc:
                print(f"❌ failed to load model '{name}': {exc}")
                return None

        if backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                print("⚠️ transformers/torch not available. Install them to use safetensors models.")
                return None

            model_dir = resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path)
            print(f"Loading transformers model '{name}' from {model_dir}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
                model = HFAutoModelForCausalLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    torch_dtype=dtype,
                )

                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

                model.to(device)
                model.eval()

                if getattr(tokenizer, "pad_token_id", None) is None:
                    tokenizer.pad_token = tokenizer.eos_token

                self.current_model = model
                self.current_tokenizer = tokenizer
                self.backend = backend
                self.current_model_config = config
                self.device = device

                print(f"✅ model '{name}' ready (backend: transformers, device: {device})")
                return model
            except Exception as exc:
                print(f"❌ failed to load transformers model '{name}': {exc}")
                self.current_model = None
                self.current_tokenizer = None
                self.device = None
                self.current_model_config = None
                self.backend = None
                return None

        print(f"❌ unsupported backend '{backend}' for model '{name}'")
        return None

    def _interpret_model_path(self, resolved_path: str) -> Tuple[str, Optional[str]]:
        """
        Handle both 'models/my-model.gguf' and directory-style layouts.
        Returns a tuple of (model_dir, model_file or None).
        """
        if os.path.isdir(resolved_path):
            return resolved_path, None

        # Single file in models/ root – pass parent dir plus explicit filename.
        directory = os.path.dirname(resolved_path)
        filename = os.path.basename(resolved_path)
        return directory or ".", filename

    def apply_prompt_template(self, prompt: str) -> str:
        if not self.current_model_config:
            return prompt

        template = self.current_model_config.get("prompt_template")
        if not template:
            return prompt

        return template.replace("{prompt}", prompt)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if not self.current_model or not self.backend:
            return "⚠️ no model loaded yet"

        text = self.apply_prompt_template(prompt)

        if self.backend == "ctransformers":
            with self._lock:
                try:
                    output = ""
                    for token in self.current_model(text, stream=True, max_new_tokens=max_tokens):
                        output += token
                    return output
                except Exception as exc:
                    return f"❌ error during generation: {exc}"

        if self.backend == "transformers":
            if not TRANSFORMERS_AVAILABLE or not self.current_tokenizer or not self.device:
                return "⚠️ transformers backend not ready"

            try:
                encoded = self.current_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                with torch.no_grad():
                    outputs = self.current_model.generate(
                        **encoded,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.current_tokenizer.pad_token_id,
                        eos_token_id=self.current_tokenizer.eos_token_id,
                    )

                generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if generated_text.startswith(text):
                    generated_text = generated_text[len(text):].strip()
                return generated_text
            except Exception as exc:
                return f"❌ transformers generation error: {exc}"

        return "⚠️ backend not supported for generation"
