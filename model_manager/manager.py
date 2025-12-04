from __future__ import annotations
import gc
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from platform_utils import is_jetson

from .backends import ASRBackend, AutoGPTQBackend, BaseBackend, DiffusersT2IBackend, HFBackend, LlamaCppBackend, MLXBackend, OCRBackend, PhiVisionBackend, TTSBackend
from .config import GenerationConfig

class ModelManager:
    SUPPORTED_EXTENSIONS = ('.gguf', '.ggml')

    def __init__(self, models_dir: str='models', history_dir: str='history', device_pref: str='auto'):
        self._models_dir = Path(models_dir)
        self._history_dir = Path(history_dir)
        try:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            self._history_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f'Permission denied creating directories: {e}') from e
        except OSError as e:
            raise RuntimeError(f'Failed to create directories: {e}') from e
        self._impl: Optional[BaseBackend] = None
        self._backend: Optional[str] = None
        self._current_model_name: Optional[str] = None
        self._history_file: Optional[Path] = None
        self._history: List[dict] = []
        self._config = GenerationConfig()
        self._history_enabled = True
        self._device_pref = device_pref
        self._llama_threads: Optional[int] = None
        self._llama_ctx: Optional[int] = 4096
        self._llama_gpu_layers: Optional[int] = self._suggest_llama_gpu_layers(device_pref)
        self._kind: str = 'text'
        self._encryptor: Optional[Any] = None
        self._generation_lock = threading.Lock()
        self._generating = False

    def set_encryptor(self, encryptor: Optional[Any]) -> None:
        self._encryptor = encryptor

    def set_history_enabled(self, enabled: bool) -> None:
        self._history_enabled = bool(enabled)

    def set_text_config(self, *, max_tokens: Optional[int]=None, temperature: Optional[float]=None) -> None:
        if max_tokens is not None:
            self._config.max_tokens = int(max_tokens)
        if temperature is not None:
            self._config.temperature = float(temperature)

    def list_models(self) -> List[str]:
        output: List[str] = []
        on_jetson = is_jetson()
        for item in sorted(self._models_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                output.append(item.name)
            elif item.is_dir():
                has_gguf = any((child.is_file() and child.suffix.lower() in self.SUPPORTED_EXTENSIONS for child in item.iterdir()))
                if has_gguf:
                    output.append(item.name)
                elif not on_jetson:
                    if (item / 'model_index.json').exists():
                        output.append(item.name)
                    elif (item / 'quantize_config.json').exists():
                        output.append(item.name)
                    elif (item / 'config.json').exists():
                        output.append(item.name)
        return output

    def _suggest_llama_gpu_layers(self, pref: str) -> Optional[int]:
        pref = (pref or 'auto').lower()
        if pref == 'cpu':
            return 0
        if pref in ('cuda', 'mps'):
            return -1
        if is_jetson():
            return 0
        try:
            import torch
            if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                return -1
        except Exception:
            pass
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                return -1
        except Exception:
            pass
        return 0
    def _detect_directory_backend(self, directory: Path) -> Optional[Tuple[str, Path, str]]:
        if is_jetson():
            first_gguf = next(
                (f for f in sorted(directory.iterdir())
                 if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS),
                None,
            )
            if first_gguf:
                return ('llama_cpp', first_gguf, 'text')
            raise ValueError('On Jetson, only GGUF/GGML models are supported. Convert this model to GGUF for llama.cpp.')
        first_gguf = next(
            (f for f in sorted(directory.iterdir())
             if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS),
            None,
        )
        if first_gguf:
            return ('llama_cpp', first_gguf, 'text')
        if (directory / 'model_index.json').exists():
            return ('diffusers_t2i', directory, 'image')
        if (directory / 'quantize_config.json').exists():
            return ('auto_gptq', directory, 'text')
        if (directory / 'adapter_config.json').exists():
            return ('transformers', directory, 'text')

        def looks_like_mlx() -> bool:
            config_path = directory / 'config.json'
            if not config_path.exists():
                return False
            try:
                config_data = json.loads(config_path.read_text())
            except Exception:
                config_data = {}
            name_lower = directory.name.lower()
            if 'mlx' in name_lower:
                return True
            quant_info = str(config_data.get('quantization_config', {})).lower()
            if 'mlx' in quant_info:
                return True
            library_name = str(config_data.get('library_name', '')).lower()
            if library_name == 'mlx':
                return True
            readme_path = directory / 'README.md'
            if readme_path.exists():
                try:
                    readme_text = readme_path.read_text(errors='ignore').lower()
                    if ('library_name: mlx' in readme_text) or ('mlx-community' in readme_text) or '\n- mlx' in readme_text:
                        return True
                except Exception:
                    pass
            index_path = directory / 'model.safetensors.index.json'
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text())
                    weight_map = index.get('weight_map') or {}
                    if any(key.endswith('.scales') for key in weight_map.keys()):
                        return True
                except Exception:
                    pass
            return False

        if looks_like_mlx():
            if is_jetson():
                raise ValueError('MLX models are not supported on Jetson. Use GGUF or Transformers models.')
            return ('mlx_lm', directory, 'text')
        config_path = directory / 'config.json'
        if not config_path.exists():
            return None
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            config = {}
        model_type = str(config.get('model_type', '')).lower()
        architectures = [a.lower() for a in config.get('architectures', [])]
        name_lower = directory.name.lower()
        quant_info = str(config.get('quantization_config', {})).lower()
        if 'mlx' in name_lower or 'mlx' in quant_info:
            if is_jetson():
                raise ValueError('MLX models are not supported on Jetson. Use GGUF or Transformers models.')
            return ('mlx_lm', directory, 'text')
        if 'phi' in name_lower and 'vision' in name_lower:
            return ('phi_vision', directory, 'vision')
        if 'vision' in model_type or 'trocr' in model_type or any('vision' in arch for arch in architectures):
            return ('ocr_trocr', directory, 'ocr')
        if 'whisper' in model_type or any('whisper' in arch for arch in architectures):
            return ('asr_whisper', directory, 'asr')
        if 'tts' in model_type or 'speech' in model_type or any('tts' in arch for arch in architectures):
            return ('tts_transformers', directory, 'tts')
        return ('transformers', directory, 'text')

    def _detect_backend(self, candidate: Path) -> Tuple[str, Path, str]:
        if candidate.is_file() and candidate.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return ('llama_cpp', candidate, 'text')
        if candidate.is_dir():
            backend = self._detect_directory_backend(candidate)
            if backend:
                return backend
        if is_jetson():
            raise ValueError('On Jetson, only GGUF/GGML models are supported. Convert the model to GGUF for llama.cpp.')
        raise ValueError(f'Unsupported model: {candidate}')

    def load_model(self, name: str, *, device_pref: Optional[str]=None) -> Tuple[bool, str]:
        candidate = self._models_dir / name
        if not candidate.exists():
            return (False, f'model not found: {candidate}')
        try:
            backend_type, model_path, kind = self._detect_backend(candidate)
        except ValueError as exc:
            return (False, str(exc))
        self._reset_session()
        target_device = device_pref or self._device_pref
        try:
            if backend_type == 'llama_cpp':
                self._impl = LlamaCppBackend(model_path, n_threads=self._llama_threads, n_ctx=self._llama_ctx, n_gpu_layers=self._llama_gpu_layers)
            elif backend_type == 'diffusers_t2i':
                self._impl = DiffusersT2IBackend(model_path, target_device)
            elif backend_type == 'auto_gptq':
                self._impl = AutoGPTQBackend(model_path, target_device)
            elif backend_type == 'transformers':
                self._impl = HFBackend(model_path, target_device)
            elif backend_type == 'ocr_trocr':
                self._impl = OCRBackend(model_path, target_device)
            elif backend_type == 'asr_whisper':
                self._impl = ASRBackend(model_path, target_device)
            elif backend_type == 'tts_transformers':
                self._impl = TTSBackend(model_path, target_device)
            elif backend_type == 'mlx_lm':
                self._impl = MLXBackend(model_path, target_device)
            elif backend_type == 'phi_vision':
                self._impl = PhiVisionBackend(model_path, target_device or self._device_pref)
            else:
                raise RuntimeError('unknown backend')
        except Exception as exc:
            self._reset_session()
            if self._looks_like_oom(exc):
                self.cleanup_memory()
                message = 'Out of memory while loading this model. Switch the device preference to CPU or pick a smaller build, then try again. Caches were cleared.'
            else:
                message = str(exc)
                if 'VisionEncoderDecoder' in message or 'image-to-text' in message:
                    message = 'This model is an OCR (image→text) model; use the OCR pipeline.'
            return (False, message)
        self._backend = self._impl.name
        self._current_model_name = name
        self._kind = kind
        self._history_file = self._history_dir / f'{self._safe_filename(name)}.json'
        self._history = self._load_history() if self._kind == 'text' and self._history_enabled else []
        return (True, str(model_path))

    def unload(self) -> None:
        if self._impl:
            try:
                self._impl.unload()
            except Exception:
                pass
        self._reset_session()
        self.cleanup_memory()

    def is_loaded(self) -> bool:
        return self._impl is not None

    def cleanup_memory(self) -> None:
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    @property
    def backend(self) -> str:
        return self._backend or 'unloaded'

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name

    def is_image_backend(self) -> bool:
        return self._kind == 'image'

    def is_ocr_backend(self) -> bool:
        return self._kind == 'ocr'

    def is_asr_backend(self) -> bool:
        return self._kind == 'asr'

    def _current_rss_mb(self) -> float:
        try:
            import psutil  # type: ignore
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        try:
            import resource  # type: ignore
            rss_kb = getattr(resource.getrusage(resource.RUSAGE_SELF), 'ru_maxrss', 0)
            if rss_kb:
                # On Linux ru_maxrss is KB; on macOS it is bytes.
                return rss_kb / 1024 if rss_kb > 1024 else rss_kb / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def run_perf_test(self, *, model_path: Optional[Path]=None, prompt: Optional[str]=None, max_tokens: int=64, n_ctx: int=1024) -> Dict[str, Any]:
        """
        Run a one-off local generation for metrics (tokens/s, latency, memory deltas).
        Uses llama.cpp (GGUF) so it works across platforms; respects Jetson CPU-only.
        """
        target = Path(model_path) if model_path else self._models_dir / 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
        if not target.exists():
            raise RuntimeError(f'Performance test model not found at {target}')
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise RuntimeError(f'llama_cpp not available: {exc}')

        test_prompt = prompt or "Say one short sentence proving this is a performance test."
        ngl = self._llama_gpu_layers if self._llama_gpu_layers is not None else -1
        n_threads = self._llama_threads if self._llama_threads else max(1, (os.cpu_count() or 1))

        stats: Dict[str, Any] = {
            'model': str(target),
            'n_ctx': n_ctx,
            'n_threads': n_threads,
            'n_gpu_layers': ngl,
            'max_tokens': max_tokens,
        }

        rss_before = self._current_rss_mb()
        stats['rss_before_mb'] = rss_before

        # Optional sampling of CPU/GPU/power while test runs.
        samples: List[Dict[str, Any]] = []
        stop_event = threading.Event()
        proc = None
        gpu_ctx: Dict[str, Any] = {}
        try:
            import psutil  # type: ignore
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(None)  # prime
        except Exception:
            proc = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_ctx = {'nvml': pynvml, 'handle': handle}
        except Exception:
            gpu_ctx = {}

        def sample_once():
            sample: Dict[str, Any] = {'ts': time.time()}
            sample['rss_mb'] = self._current_rss_mb()
            if proc:
                try:
                    import psutil  # type: ignore
                    sample['cpu_proc_pct'] = proc.cpu_percent(interval=None)
                    sample['cpu_sys_pct'] = psutil.cpu_percent(interval=None)
                except Exception:
                    pass
            if gpu_ctx:
                try:
                    nvml = gpu_ctx['nvml']
                    handle = gpu_ctx['handle']
                    mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                    sample['vram_mb'] = mem.used / (1024 * 1024)
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(handle)
                        sample['power_w'] = power / 1000.0
                    except Exception:
                        pass
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        sample['temp_c'] = temp
                    except Exception:
                        pass
                except Exception:
                    pass
            samples.append(sample)

        def sampler():
            while not stop_event.is_set():
                sample_once()
                stop_event.wait(0.5)

        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()

        t0 = time.time()
        llama = Llama(
            model_path=str(target),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=ngl,
            verbose=False,
        )
        load_s = time.time() - t0

        t1 = time.time()
        result = llama(
            test_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=None,
            echo=False,
        )
        infer_s = time.time() - t1

        # Cleanup
        try:
            del llama
        except Exception:
            pass
        self.cleanup_memory()
        rss_after = self._current_rss_mb()
        stop_event.set()
        sampler_thread.join(timeout=1.0)

        usage = result.get('usage', {}) if isinstance(result, dict) else {}
        timings = result.get('timings', {}) if isinstance(result, dict) else {}
        completion_tokens = usage.get('completion_tokens') or usage.get('completion_tokens', 0)
        prompt_tokens = usage.get('prompt_tokens')
        total_tokens = usage.get('total_tokens')

        def _agg(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {'min': None, 'max': None, 'avg': None}
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
            }

        cpu_proc = _agg([s.get('cpu_proc_pct') for s in samples if isinstance(s.get('cpu_proc_pct'), (int, float))])
        cpu_sys = _agg([s.get('cpu_sys_pct') for s in samples if isinstance(s.get('cpu_sys_pct'), (int, float))])
        vram = _agg([s.get('vram_mb') for s in samples if isinstance(s.get('vram_mb'), (int, float))])
        power = _agg([s.get('power_w') for s in samples if isinstance(s.get('power_w'), (int, float))])
        temp = _agg([s.get('temp_c') for s in samples if isinstance(s.get('temp_c'), (int, float))])

        stats.update({
            'load_s': load_s,
            'infer_s': infer_s,
            'total_s': load_s + infer_s,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'prompt_tps': timings.get('prompt_per_second') or timings.get('prompt_throughput'),
            'eval_tps': timings.get('eval_per_second') or timings.get('eval_throughput'),
            'rss_after_mb': rss_after,
            'rss_delta_mb': (rss_after - rss_before) if (rss_after and rss_before) else None,
            'cpu_proc_pct': cpu_proc,
            'cpu_sys_pct': cpu_sys,
            'vram_mb': vram,
            'power_w': power,
            'temp_c': temp,
        })

        # Fallback TPS if timings are missing
        if completion_tokens and infer_s > 0 and not stats.get('eval_tps'):
            stats['eval_tps'] = completion_tokens / infer_s

        return stats

    def is_tts_backend(self) -> bool:
        return self._kind == 'tts'

    def is_vision_backend(self) -> bool:
        return self._kind == 'vision'

    def generate(self, user_prompt: str) -> str:
        if not self._impl:
            raise RuntimeError('No model loaded.')
        if self._kind != 'text':
            raise RuntimeError('Loaded model is not a text generator.')
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError('Generation already in progress. Please wait for the current request to complete.')
        try:
            self._generating = True
            snapshot = list(self._history)
            self._history.append({'role': 'user', 'content': user_prompt})
            use_template = hasattr(self._impl, 'format_chat')
            stop_backup: Optional[Tuple[str, ...]] = None
            prompt: str
            if use_template:
                prompt = self._format_with_template()
                if prompt:
                    stop_backup = self._config.stop
                    self._config.stop = ()
                else:
                    use_template = False
            if not use_template:
                prompt = self._build_prompt_plain() if self._history_enabled else user_prompt
            try:
                text = self._impl.generate(prompt, self._config)
                text = self._postprocess_response(text)
                if self._history_enabled:
                    self._history.append({'role': 'assistant', 'content': text})
                    self._save_history()
                else:
                    self._history = snapshot
                return text
            except Exception as exc:
                self._history = snapshot
                if self._looks_like_oom(exc):
                    self.cleanup_memory()
                    raise RuntimeError('Generation failed: out of memory. Lower max tokens, disable chat history, or run Free VRAM from Settings.') from exc
                raise RuntimeError(f'Generation failed: {exc}') from exc
            finally:
                if stop_backup is not None:
                    self._config.stop = stop_backup
        finally:
            self._generating = False
            self._generation_lock.release()

    def run_ocr(self, image_path: str) -> str:
        if not self._impl or not self.is_ocr_backend():
            raise RuntimeError('No OCR model loaded.')
        runner = getattr(self._impl, 'run', None)
        if not callable(runner):
            raise RuntimeError('Loaded model does not support OCR.')
        return runner(image_path)

    def run_asr(self, audio_path: str) -> str:
        if not self._impl or not self.is_asr_backend():
            raise RuntimeError('No ASR model loaded.')
        runner = getattr(self._impl, 'run', None)
        if not callable(runner):
            raise RuntimeError('Loaded model does not support ASR.')
        return runner(audio_path)

    def run_tts(self, text: str, outdir: Path) -> str:
        if not self._impl or not self.is_tts_backend():
            raise RuntimeError('No TTS model loaded.')
        runner = getattr(self._impl, 'run', None)
        if not callable(runner):
            raise RuntimeError('Loaded model does not support TTS.')
        return runner(text, outdir)

    def generate_image(self, prompt: str, **kwargs) -> str:
        if not self._impl:
            raise RuntimeError('No model loaded.')
        if not hasattr(self._impl, 'generate_image'):
            raise RuntimeError('Loaded model does not support image generation.')
        return self._impl.generate_image(prompt, **kwargs)

    def analyze_image(self, image_path: str, question: str) -> str:
        if not self._impl or not self.is_vision_backend():
            raise RuntimeError('No vision model loaded.')
        analyzer = getattr(self._impl, 'analyze_image', None)
        if not callable(analyzer):
            raise RuntimeError('Loaded model does not support image analysis.')
        return analyzer(Path(image_path), question, self._config)

    def get_history(self) -> List[dict]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history = []
        if self._history_file and self._history_file.exists():
            try:
                self._history_file.unlink()
            except OSError:
                pass

    def add_history_entry(self, role: str, content: str) -> None:
        if self._kind != 'text':
            return
        entry = {'role': role, 'content': content}
        self._history.append(entry)
        if self._history_enabled:
            self._save_history()

    def _reset_session(self) -> None:
        self._impl = None
        self._backend = None
        self._current_model_name = None
        self._history_file = None
        self._history = []
        self._kind = 'text'

    def describe_session(self) -> str:
        if not self._impl or not self._current_model_name:
            return 'No model loaded.'
        details = [f"Model: {self._current_model_name}", f"Backend: {self._backend or 'unknown'}"]
        inspector = getattr(self._impl, 'runtime_info', None)
        runtime: dict[str, Any] = {}
        if callable(inspector):
            try:
                runtime = inspector() or {}
            except Exception:
                runtime = {}
        device = runtime.get('device') or self._device_pref or 'cpu'
        details.append(f'Device: {device}')
        dtype = runtime.get('dtype')
        if dtype:
            details.append(f'Precision: {dtype}')
        context = runtime.get('max_position_embeddings') or runtime.get('tokenizer_max_length')
        if context:
            details.append(f'Context limit: {context} tokens')
        details.append(f'Max tokens: {self._config.max_tokens}')
        details.append(f'Temperature: {self._config.temperature}')
        details.append(f"History: {'on' if self._history_enabled else 'off'}")
        return '\n'.join(details)

    def _looks_like_oom(self, exc: Exception) -> bool:
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        message = str(exc).lower()
        if not message:
            return False
        for token in ('out of memory', 'mps backend', 'unable to allocate', 'cuda error 2'):
            if token in message:
                return True
        return False

    def _load_history(self) -> List[dict]:
        if not self._history_file or not self._history_file.exists():
            return []
        try:
            blob = self._history_file.read_bytes()
        except Exception:
            return []
        if not blob:
            return []
        if self._encryptor:
            try:
                plaintext = self._encryptor.decrypt(blob)
                data = json.loads(plaintext.decode('utf-8'))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            try:
                fallback_text = blob.decode('utf-8')
                data = json.loads(fallback_text)
                if isinstance(data, list):
                    try:
                        payload = self._encryptor.encrypt(fallback_text.encode('utf-8'))
                        self._history_file.write_bytes(payload)
                    except Exception:
                        pass
                    return data
            except Exception:
                return []
            return []
        try:
            text = blob.decode('utf-8')
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    def _save_history(self) -> None:
        if not self._history_file:
            return
        try:
            payload = json.dumps(self._history, ensure_ascii=False, indent=2).encode('utf-8')
            if self._encryptor:
                payload = self._encryptor.encrypt(payload)
                self._history_file.write_bytes(payload)
            else:
                self._history_file.write_text(payload.decode('utf-8'), encoding='utf-8')
        except Exception:
            pass

    def _build_prompt_plain(self) -> str:
        parts = []
        for message in self._history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'user':
                parts.append(f'User: {content}')
            elif role == 'assistant':
                parts.append(f'Assistant: {content}')
        parts.append('Assistant:')
        return '\n'.join(parts)

    def _safe_filename(self, name: str) -> str:
        return re.sub('[^A-Za-z0-9_.-]', '_', name)

    def _format_with_template(self) -> str:
        formatter = getattr(self._impl, 'format_chat', None)
        if not callable(formatter):
            return ''
        try:
            formatted = formatter(self._history)
        except Exception:
            return ''
        return formatted or ''

    def _postprocess_response(self, text: str) -> str:
        cleaned = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        cleaned = re.sub('<[^>]+>', '', cleaned)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return '\n'.join(lines)
