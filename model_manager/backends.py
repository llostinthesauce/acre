from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf
import torch
from PIL import Image

from .config import GenerationConfig


class BaseBackend:
    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        raise NotImplementedError

    def generate_image(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


def pick_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    try:
        import torch

        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_mps = False
        has_cuda = False
    if pref == "mps" and has_mps:
        return "mps"
    if pref == "cuda" and has_cuda:
        return "cuda"
    if pref == "cpu":
        return "cpu"
    if has_mps:
        return "mps"
    if has_cuda:
        return "cuda"
    return "cpu"


class LlamaCppBackend(BaseBackend):
    def __init__(
        self,
        model_path: Path,
        n_threads: Optional[int] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):
        try:
            from llama_cpp import Llama
        except Exception:
            raise RuntimeError("llama_cpp not installed")
        kwargs: dict[str, int | str] = {"model_path": str(model_path)}
        if n_threads and n_threads > 0:
            kwargs["n_threads"] = int(n_threads)
        if n_ctx and n_ctx > 0:
            kwargs["n_ctx"] = int(n_ctx)
        if n_gpu_layers is not None and n_gpu_layers >= 0:
            kwargs["n_gpu_layers"] = int(n_gpu_layers)
        self._llama = Llama(**kwargs)

    @property
    def name(self) -> str:
        return "llama_cpp"

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        output = self._llama(
            prompt_text,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=cfg.stop if cfg.stop else None,
            echo=False,
        )
        text = output["choices"][0]["text"]
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        self._llama = None


class HFBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            raise RuntimeError("transformers and torch required")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        import torch

        self._device = pick_device(device_pref)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, dtype="auto"
        )
        if self._device != "cpu":
            self._model.to(self._device)

    @property
    def name(self) -> str:
        return "transformers"

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        import torch

        encoded = self._tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )
        if self._device != "cpu":
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        max_ctx = getattr(
            self._model.config,
            "max_position_embeddings",
            getattr(self._tokenizer, "model_max_length", 2048),
        )
        input_len = int(encoded["input_ids"].shape[1])
        desired_new = int(cfg.max_tokens)
        if input_len + desired_new > max_ctx:
            keep = max(1, max_ctx - desired_new)
            encoded["input_ids"] = encoded["input_ids"][:, -keep:]
            if "attention_mask" in encoded:
                encoded["attention_mask"] = encoded["attention_mask"][:, -keep:]
            input_len = int(encoded["input_ids"].shape[1])
        new_tokens = max(1, min(desired_new, max_ctx - input_len))
        do_sample = cfg.temperature > 0.0
        try:
            generated = self._model.generate(
                **encoded,
                max_new_tokens=new_tokens,
                do_sample=do_sample,
                temperature=float(cfg.temperature) if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception as exc:
            try:
                self._model.to("cpu")
                encoded = {k: v.to("cpu") for k, v in encoded.items()}
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            except Exception as second:
                raise RuntimeError(
                    "Transformers generation failed; try lowering tokens or temperature"
                ) from second
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text) :]
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        try:
            import torch

            del self._model
            del self._tokenizer
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def format_chat(self, messages: List[dict]) -> str:
        template = getattr(self._tokenizer, "chat_template", None)
        if not template:
            return ""
        enable_thinking: Optional[bool] = None
        for item in messages:
            content = str(item.get("content", ""))
            if "/no_think" in content:
                enable_thinking = False
            elif "/think" in content:
                enable_thinking = True
        kwargs = dict(messages=messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is None:
            kwargs["enable_thinking"] = False
        else:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self._tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return self._tokenizer.apply_chat_template(**kwargs)
            except Exception:
                return ""
        except Exception:
            return ""


class AutoGPTQBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except Exception:
            raise RuntimeError("auto-gptq, transformers and torch required")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True
        )
        self._device = pick_device(device_pref)
        device_map = "auto" if self._device != "cpu" else None
        self._model = AutoGPTQForCausalLM.from_quantized(
            str(model_dir),
            device_map=device_map,
            dtype="auto",
            use_safetensors=True,
            local_files_only=True,
        )
        if self._device == "mps":
            self._model.to("mps")

    @property
    def name(self) -> str:
        return "auto_gptq"

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        if self._device != "cpu":
            import torch

            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        try:
            generated = self._model.generate(
                **encoded,
                max_new_tokens=cfg.max_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception as exc:
            raise RuntimeError(f"AutoGPTQ generation failed: {exc}") from exc
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text) :]
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        try:
            import torch

            del self._model
            del self._tokenizer
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass


class MLXBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from mlx_lm import load as mlx_load
        except Exception:
            raise RuntimeError("mlx_lm required for MLX quantized models")
        self._model, self._tokenizer = mlx_load(str(model_dir))

    @property
    def name(self) -> str:
        return "mlx_lm"

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        try:
            from mlx_lm import generate as mlx_generate
        except Exception as exc:
            raise RuntimeError(f"mlx_lm.generate unavailable: {exc}") from exc
        max_tokens = int(max(1, cfg.max_tokens))
        temperature = float(cfg.temperature if cfg.temperature > 0 else 0.01)
        kwargs = dict(prompt=prompt_text, max_tokens=max_tokens, temperature=temperature, verbose=False)
        try:
            result = mlx_generate(self._model, self._tokenizer, **kwargs)
        except TypeError:
            kwargs.pop("temperature", None)
            result = mlx_generate(self._model, self._tokenizer, **kwargs)
        text = result if isinstance(result, str) else str(result)
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token and stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def format_chat(self, messages: List[dict]) -> str:
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return ""
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            return ""
        enable_thinking: Optional[bool] = None
        for item in messages:
            content = str(item.get("content", ""))
            if "/no_think" in content:
                enable_thinking = False
            elif "/think" in content:
                enable_thinking = True
        kwargs = dict(messages=messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is None:
            kwargs["enable_thinking"] = False
        else:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return tokenizer.apply_chat_template(**kwargs)
            except Exception:
                return ""
        except Exception:
            return ""


class PhiVisionBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        except Exception:
            raise RuntimeError("transformers and torch required for vision models")
        self._device = pick_device(device_pref)
        torch_dtype = torch.float16 if self._device in {"cuda", "mps"} else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        img_proc = getattr(self._processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "num_crops"):
            try:
                img_proc.num_crops = 1
            except Exception:
                try:
                    setattr(img_proc, "num_crops", 1)
                except Exception:
                    pass
        config = AutoConfig.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        for attr in (
            "use_flash_attn",
            "use_flash_attention",
            "use_flash_attention_2",
            "enable_flash_attn_2",
            "enable_flash_attention_2",
        ):
            if hasattr(config, attr):
                setattr(config, attr, False)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        if hasattr(config, "use_cache"):
            config.use_cache = False
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch_dtype,
            trust_remote_code=True,
            config=config,
            attn_implementation="eager",
        )
        if self._device != "cpu":
            self._model.to(self._device)
        self._model.eval()
        self._apply_template = getattr(self._processor, "apply_chat_template", None)

    def _build_prompt(self, question: str, *, include_image: bool) -> str:
        user_prefix = "<|user|>\n"
        if include_image:
            user_prefix += "<|image_1|>\n"
        if callable(self._apply_template):
            content = ""
            if include_image:
                content += "<|image_1|>\n"
            content += question
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            try:
                prompt = self._apply_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if prompt.endswith("<|endoftext|>"):
                    prompt = prompt.rsplit("<|endoftext|>", 1)[0]
                return prompt
            except Exception:
                pass
        return f"{user_prefix}{question}<|end|>\n<|assistant|>\n"

    @property
    def name(self) -> str:
        return "phi_vision"

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        prompt = self._build_prompt(prompt_text, include_image=False)
        inputs = self._processor(text=prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        max_new = max(1, min(int(cfg.max_tokens), 160))
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                use_cache=False,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return text[0].strip()

    def analyze_image(self, image_path: Path, prompt: str, cfg: GenerationConfig) -> str:
        image = Image.open(image_path).convert("RGB")
        max_side = max(image.size)
        if max_side > 1024:
            scale = 1024 / max_side
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.BICUBIC)
        chat_prompt = self._build_prompt(prompt, include_image=True)
        inputs = self._processor(text=chat_prompt, images=[image], return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        max_new = max(64, min(int(cfg.max_tokens), 160))
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=min(max_new, 256),
                do_sample=False,
                temperature=None,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                use_cache=False,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return text[0].strip()

    def unload(self) -> None:
        try:
            import torch

            del self._model
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        self._processor = None


class DiffusersT2IBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from diffusers import AutoPipelineForText2Image
        except Exception:
            raise RuntimeError("diffusers, transformers and torch required")
        self._device = pick_device(device_pref)
        from diffusers import AutoPipelineForText2Image

        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            str(model_dir), local_files_only=True, dtype="auto", use_safetensors=True
        )
        try:
            self._pipeline.safety_checker = None
        except Exception:
            pass
        self._pipeline = self._pipeline.to(self._device)

    @property
    def name(self) -> str:
        return "diffusers_t2i"

    def _align(self, value: int) -> int:
        return max(8, int(value // 8 * 8))

    def generate_image(
        self,
        prompt: str,
        steps: int = 4,
        guidance: float = 0.0,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        outdir: Path = Path("outputs"),
    ) -> str:
        import torch

        outdir.mkdir(parents=True, exist_ok=True)
        width = self._align(width)
        height = self._align(height)
        steps = max(1, min(steps, 50))
        generator = torch.Generator(device=self._device)
        if seed is None:
            seed = int(time.time()) & 0xFFFFFFFF
        generator = generator.manual_seed(int(seed))
        try:
            result = self._pipeline(
                prompt,
                num_inference_steps=steps,
                guidance_scale=float(guidance),
                width=width,
                height=height,
                generator=generator,
            )
            image = result.images[0]
        except Exception as exc:
            if "embedding images is forbidden" in str(exc).lower():
                raise RuntimeError(
                    "This diffusion model only supports pure text prompts (no init image)."
                ) from exc
            raise RuntimeError(f"Diffusers generation failed: {exc}") from exc
        path = outdir / f"img_{seed}.png"
        image.save(path)
        return str(path)

    def unload(self) -> None:
        try:
            import torch

            self._pipeline = None
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass


class OCRBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError("transformers (vision) required for OCR")
        self._pipeline = pipeline(
            "image-to-text",
            model=str(model_dir),
            tokenizer=None,
            feature_extractor=None,
            device=-1,
            local_files_only=True,
        )

    @property
    def name(self) -> str:
        return "ocr_trocr"

    def run(self, image_path: str) -> str:
        try:
            outputs = self._pipeline(image_path)
            if isinstance(outputs, list) and outputs and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]
            return str(outputs)
        except Exception as exc:
            raise RuntimeError(f"OCR failed: {exc}") from exc

    def unload(self) -> None:
        self._pipeline = None


class ASRBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError("transformers (audio) required for ASR")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=str(model_dir),
            device=-1,
            local_files_only=True,
        )

    @property
    def name(self) -> str:
        return "asr_whisper"

    def run(self, audio_path: str) -> str:
        try:
            output = self._pipeline(audio_path)
            if isinstance(output, dict) and "text" in output:
                return output["text"]
            return str(output)
        except Exception as exc:
            raise RuntimeError(f"ASR failed: {exc}") from exc

    def unload(self) -> None:
        self._pipeline = None


class TTSBackend(BaseBackend):
    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError("transformers (TTS) required for text-to-speech")
        self._pipeline = pipeline(
            "text-to-speech", model=str(model_dir), device=-1, local_files_only=True
        )

    @property
    def name(self) -> str:
        return "tts_transformers"

    def run(self, text: str, outdir: Path) -> str:
        try:
            audio = self._pipeline(text)
            samples = audio["audio"]
            sample_rate = int(audio.get("sampling_rate", 22050))
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"tts_{int(time.time())}.wav"
            sf.write(str(path), samples, sample_rate)
            return str(path)
        except Exception as exc:
            raise RuntimeError(f"TTS failed: {exc}") from exc

    def unload(self) -> None:
        self._pipeline = None
