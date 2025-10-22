from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    stop: Tuple[str, ...] = ("User:",)

class _BaseBackend:
    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str: ...
    def unload(self) -> None: ...
    @property
    def name(self) -> str: ...

class _LlamaCppBackend(_BaseBackend):
    def __init__(self, model_path: Path):
        try:
            from llama_cpp import Llama
        except Exception:
            raise RuntimeError("llama_cpp not installed")
        self._ll = Llama(model_path=str(model_path))
    @property
    def name(self) -> str:
        return "llama_cpp"
    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        out = self._ll(
            prompt_text,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=cfg.stop if cfg.stop else None,
            echo=False,
        )
        txt = out["choices"][0]["text"]
        if cfg.stop:
            for s in cfg.stop:
                if s in txt:
                    txt = txt.split(s, 1)[0]
        return txt.strip()
    def unload(self) -> None:
        self._ll = None

class _HFBackend(_BaseBackend):
    def __init__(self, model_dir: Path):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            raise RuntimeError("transformers and torch required")
        self._tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self._device = "cpu"
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
        except Exception:
            self._device = "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype="auto"
        )
        if self._device != "cpu":
            self._model.to(self._device)
    @property
    def name(self) -> str:
        return "transformers"
    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        import torch
        enc = self._tok(prompt_text, return_tensors="pt")
        if self._device != "cpu":
            enc = {k: v.to(self._device) for k, v in enc.items()}
        gen = self._model.generate(
            **enc,
            max_new_tokens=cfg.max_tokens,
            do_sample=(cfg.temperature > 0),
            temperature=cfg.temperature,
            pad_token_id=self._tok.eos_token_id,
            eos_token_id=self._tok.eos_token_id,
        )
        txt = self._tok.decode(gen[0], skip_special_tokens=True)
        if prompt_text and txt.startswith(prompt_text):
            txt = txt[len(prompt_text):]
        if cfg.stop:
            for s in cfg.stop:
                if s in txt:
                    txt = txt.split(s, 1)[0]
        return txt.strip()
    def unload(self) -> None:
        self._model = None
        self._tok = None

class _AutoGPTQBackend(_BaseBackend):
    def __init__(self, model_dir: Path):
        try:
            import torch  # noqa: F401
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except Exception:
            raise RuntimeError("auto-gptq, transformers and torch required")
        self._tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
        self._device = "cpu"
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
        except Exception:
            self._device = "cpu"
        self._model = AutoGPTQForCausalLM.from_quantized(
            str(model_dir),
            device_map="auto" if self._device != "cpu" else None,
            torch_dtype="auto",
            use_safetensors=True,
            local_files_only=True,
        )
        if self._device == "mps":
            self._model.to("mps")
    @property
    def name(self) -> str:
        return "auto_gptq"
    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        import torch
        enc = self._tok(prompt_text, return_tensors="pt")
        if self._device != "cpu":
            enc = {k: v.to(self._device) for k, v in enc.items()}
        gen = self._model.generate(
            **enc,
            max_new_tokens=cfg.max_tokens,
            do_sample=(cfg.temperature > 0),
            temperature=cfg.temperature,
            pad_token_id=self._tok.eos_token_id,
            eos_token_id=self._tok.eos_token_id,
        )
        txt = self._tok.decode(gen[0], skip_special_tokens=True)
        if prompt_text and txt.startswith(prompt_text):
            txt = txt[len(prompt_text):]
        if cfg.stop:
            for s in cfg.stop:
                if s in txt:
                    txt = txt.split(s, 1)[0]
        return txt.strip()
    def unload(self) -> None:
        self._model = None
        self._tok = None

class ModelManager:
    SUPPORTED_EXTENSIONS = (".gguf",)

    def __init__(self, models_dir: str = "models", history_dir: str = "history"):
        self._models_dir = Path(models_dir)
        self._history_dir = Path(history_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._impl: Optional[_BaseBackend] = None
        self._backend: Optional[str] = None
        self._current_model_name: Optional[str] = None
        self._history_file: Optional[Path] = None
        self._history: List[dict] = []
        self._config = GenerationConfig()

    def list_models(self) -> List[str]:
        out: List[str] = []
        for item in sorted(self._models_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                out.append(item.name)
            elif item.is_dir():
                if any(ch.is_file() and ch.suffix.lower() == ".gguf" for ch in item.iterdir()):
                    out.append(item.name)
                elif (item / "quantize_config.json").exists():
                    out.append(item.name)
                elif (item / "config.json").exists():
                    out.append(item.name)
        return out

    def _detect_backend(self, candidate: Path) -> tuple[str, Path]:
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return ("llama_cpp", candidate)
        if candidate.is_dir():
            gguf = sorted(ch for ch in candidate.iterdir() if ch.is_file() and ch.suffix.lower() == ".gguf")
            if gguf:
                return ("llama_cpp", gguf[0])
            if (candidate / "quantize_config.json").exists():
                return ("auto_gptq", candidate)
            if (candidate / "config.json").exists():
                return ("transformers", candidate)
        raise ValueError(f"Unsupported model location: {candidate}")

    def load_model(self, name: str) -> bool:
        cand = self._models_dir / name
        if not cand.exists():
            print(f"❌ model not found: {cand}")
            return False
        try:
            btype, mpath = self._detect_backend(cand)
        except ValueError as exc:
            print(f"❌ {exc}")
            return False
        self._reset_session()
        try:
            if btype == "llama_cpp":
                self._impl = _LlamaCppBackend(mpath)
            elif btype == "auto_gptq":
                self._impl = _AutoGPTQBackend(mpath)
            elif btype == "transformers":
                self._impl = _HFBackend(mpath)
            else:
                raise RuntimeError("unknown backend")
        except Exception as exc:
            print(f"❌ failed to load model: {exc}")
            self._reset_session()
            return False
        self._backend = self._impl.name
        self._current_model_name = name
        self._history_file = self._history_dir / f"{self._safe_filename(name)}.json"
        self._history = self._load_history()
        print(f"✅ model loaded: {mpath}")
        return True

    def unload(self) -> None:
        if self._impl:
            try:
                self._impl.unload()
            except Exception:
                pass
        self._reset_session()

    def is_loaded(self) -> bool:
        return self._impl is not None

    @property
    def backend(self) -> str:
        return self._backend or "unloaded"

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name

    def generate(self, user_prompt: str) -> str:
        if not self._impl:
            raise RuntimeError("No model loaded.")
        snap = list(self._history)
        self._history.append({"role": "user", "content": user_prompt})
        prompt = self._build_prompt()
        try:
            text = self._impl.generate(prompt, self._config)
            self._history.append({"role": "assistant", "content": text})
            self._save_history()
            return text
        except Exception as exc:
            self._history = snap
            raise RuntimeError(f"Generation failed: {exc}") from exc

    def get_history(self) -> List[dict]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history = []
        if self._history_file and self._history_file.exists():
            try:
                self._history_file.unlink()
            except OSError:
                pass

    def _reset_session(self) -> None:
        self._impl = None
        self._backend = None
        self._current_model_name = None
        self._history_file = None
        self._history = []

    def _load_history(self) -> List[dict]:
        if not self._history_file or not self._history_file.exists():
            return []
        try:
            with self._history_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_history(self) -> None:
        if not self._history_file:
            return
        try:
            with self._history_file.open("w", encoding="utf-8") as fh:
                json.dump(self._history, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_prompt(self) -> str:
        parts = []
        for m in self._history:
            r = m.get("role", "user")
            c = m.get("content", "")
            if r == "user":
                parts.append(f"User: {c}")
            elif r == "assistant":
                parts.append(f"Assistant: {c}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _safe_filename(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)