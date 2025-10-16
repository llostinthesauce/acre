from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - surfaced via UI
    Llama = None  # type: ignore[misc]


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    stop: Tuple[str, ...] = ("User:",)


class ModelManager:
    """Handles model discovery, loading, inference, and chat history tracking."""

    SUPPORTED_EXTENSIONS = (".gguf",)

    def __init__(self, models_dir: str = "models", history_dir: str = "history"):
        self._models_dir = Path(models_dir)
        self._history_dir = Path(history_dir)

        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)

        self._llama: Optional[Llama] = None
        self._backend: Optional[str] = None
        self._current_model_name: Optional[str] = None
        self._history_file: Optional[Path] = None
        self._history: List[dict] = []
        self._config = GenerationConfig()

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def list_models(self) -> List[str]:
        entries: List[str] = []
        for item in sorted(self._models_dir.iterdir()):
            if self._is_supported_file(item):
                entries.append(item.name)
            elif item.is_dir() and any(self._is_supported_file(child) for child in item.iterdir()):
                entries.append(item.name)
        return entries

    def load_model(self, name: str) -> bool:
        candidate = self._models_dir / name
        if not candidate.exists():
            print(f"❌ model not found: {candidate}")
            return False

        try:
            model_path = self._resolve_model_path(candidate)
        except ValueError as exc:
            print(f"❌ {exc}")
            return False

        if Llama is None:
            print("⚠️ llama_cpp not installed.")
            return False

        self._reset_session()

        try:
            self._llama = Llama(model_path=str(model_path))
        except Exception as exc:  # pragma: no cover - runtime feedback
            print(f"❌ failed to load model: {exc}")
            self._reset_session()
            return False

        self._backend = "llama_cpp"
        self._current_model_name = name
        self._history_file = self._history_dir / f"{self._safe_filename(name)}.json"
        self._history = self._load_history()
        print(f"✅ model loaded: {model_path}")
        return True

    def unload(self) -> None:
        self._reset_session()

    def is_loaded(self) -> bool:
        return self._llama is not None

    @property
    def backend(self) -> str:
        return self._backend or "unloaded"

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name

    def generate(self, user_prompt: str) -> str:
        if not self._llama:
            raise RuntimeError("No model loaded.")

        snapshot = list(self._history)
        self._history.append({"role": "user", "content": user_prompt})
        prompt_text = self._build_prompt()

        try:
            output = self._llama(
                prompt_text,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                stop=self._config.stop if self._config.stop else None,
                echo=False,
            )
            assistant_text = output["choices"][0]["text"]
            if self._config.stop:
                for stopper in self._config.stop:
                    if stopper in assistant_text:
                        assistant_text = assistant_text.split(stopper, 1)[0]
            assistant_text = assistant_text.strip()
            self._history.append({"role": "assistant", "content": assistant_text})
            self._save_history()
            return assistant_text
        except Exception as exc:
            self._history = snapshot
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

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _reset_session(self) -> None:
        self._llama = None
        self._backend = None
        self._current_model_name = None
        self._history_file = None
        self._history = []

    def _is_supported_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _resolve_model_path(self, candidate: Path) -> Path:
        if candidate.is_file():
            if not self._is_supported_file(candidate):
                raise ValueError(f"Unsupported file type: {candidate.suffix}")
            return candidate

        if candidate.is_dir():
            gguf_files = sorted(child for child in candidate.iterdir() if self._is_supported_file(child))
            if gguf_files:
                return gguf_files[0]
            raise ValueError(f"No .gguf file found in directory: {candidate}")

        raise ValueError(f"Invalid model path: {candidate}")

    def _load_history(self) -> List[dict]:
        if not self._history_file or not self._history_file.exists():
            return []
        try:
            with self._history_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception as exc:
            print(f"⚠️ failed to load history {self._history_file}: {exc}")
        return []

    def _save_history(self) -> None:
        if not self._history_file:
            return
        try:
            with self._history_file.open("w", encoding="utf-8") as fh:
                json.dump(self._history, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"⚠️ failed to save history: {exc}")

    def _build_prompt(self) -> str:
        parts = []
        for message in self._history:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _safe_filename(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
