from __future__ import annotations
import gc
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import soundfile as sf
from PIL import Image
from platform_utils import is_jetson
try:
    import torch
except ImportError:
    torch = None
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

    def runtime_info(self) -> Dict[str, Any]:
        return {}

def pick_device(pref: str) -> str:
    pref = (pref or 'auto').lower()
    if is_jetson():
        return 'cpu'
    try:
        import torch
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_mps = False
        has_cuda = False
    if pref == 'mps' and has_mps:
        return 'mps'
    if pref == 'cuda' and has_cuda:
        return 'cuda'
    if pref == 'cpu':
        return 'cpu'
    if has_mps:
        return 'mps'
    if has_cuda:
        return 'cuda'
    return 'cpu'

class LlamaCppBackend(BaseBackend):

    def __init__(self, model_path: Path, n_threads: Optional[int]=None, n_ctx: Optional[int]=None, n_gpu_layers: Optional[int]=None):
        try:
            from llama_cpp import Llama
        except Exception:
            raise RuntimeError('llama_cpp not installed')
        kwargs: dict[str, int | str] = {'model_path': str(model_path)}
        if n_threads and n_threads > 0:
            kwargs['n_threads'] = int(n_threads)
        if n_ctx and n_ctx > 0:
            kwargs['n_ctx'] = int(n_ctx)
        if n_gpu_layers is not None:
            kwargs['n_gpu_layers'] = int(n_gpu_layers)
        self._llama = Llama(**kwargs)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return 'llama_cpp'

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        with self._lock:
            if self._llama is None:
                raise RuntimeError('Model has been unloaded')
            output = self._llama(prompt_text, max_tokens=cfg.max_tokens, temperature=cfg.temperature, stop=cfg.stop if cfg.stop else None, echo=False)
            text = output['choices'][0]['text']
            if cfg.stop:
                for stop_token in cfg.stop:
                    if stop_token in text:
                        text = text.split(stop_token, 1)[0]
            return text.strip()

    def unload(self) -> None:
        with self._lock:
            self._llama = None

class HFBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for transformers backends. Please install PyTorch for Jetson.')
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            raise RuntimeError('transformers and torch required')
        import torch as torch_module
        trust_remote = self._should_trust_remote_code(model_dir)
        tokenizer_kwargs: dict[str, Any] = {'pretrained_model_name_or_path': str(model_dir), 'local_files_only': True}
        model_kwargs: dict[str, Any] = {'pretrained_model_name_or_path': str(model_dir), 'local_files_only': True, 'torch_dtype': 'auto', 'low_cpu_mem_usage': True}
        if trust_remote:
            tokenizer_kwargs['trust_remote_code'] = True
            model_kwargs['trust_remote_code'] = True
        self._tokenizer = self._load_tokenizer(AutoTokenizer, tokenizer_kwargs, model_dir)
        self._device = pick_device(device_pref)
        adapter_config_path = model_dir / 'adapter_config.json'
        if adapter_config_path.exists():
            self._model = self._load_lora_model(model_dir, model_kwargs, trust_remote)
        else:
            self._model = self._load_model(AutoModelForCausalLM, model_kwargs)
        if self._device != 'cpu':
            self._model.to(self._device)
        self._model.eval()
        config = getattr(self._model, 'config', None)
        if config is not None and (not hasattr(config, 'num_hidden_layers')):
            candidate = getattr(config, 'num_transformer_layers', None)
            if isinstance(candidate, int) and candidate > 0:
                try:
                    setattr(config, 'num_hidden_layers', candidate)
                except Exception:
                    pass
        if config is not None and hasattr(config, 'use_cache'):
            try:
                setattr(config, 'use_cache', False)
            except Exception:
                pass
        gen_conf = getattr(self._model, 'generation_config', None)
        if gen_conf is not None:
            try:
                setattr(gen_conf, 'use_cache', False)
            except Exception:
                pass
            try:
                setattr(gen_conf, 'cache_implementation', 'static')
            except Exception:
                pass
        if getattr(self._tokenizer, 'pad_token_id', None) is None and getattr(self._tokenizer, 'eos_token', None):
            try:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception:
                pass

    def _load_lora_model(self, model_dir: Path, model_kwargs: Dict[str, Any], trust_remote: bool) -> Any:
        import json
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise RuntimeError('PEFT library is required to load LoRA fine-tuned models. Install with: pip install peft')
        adapter_config_path = model_dir / 'adapter_config.json'
        try:
            adapter_config = json.loads(adapter_config_path.read_text())
            base_model_path_str = adapter_config.get('base_model_name_or_path')
        except Exception as exc:
            raise RuntimeError(f'Failed to read adapter_config.json: {exc}') from exc
        if not base_model_path_str:
            raise RuntimeError("adapter_config.json missing 'base_model_name_or_path'")
        base_model_path = Path(base_model_path_str)
        if not base_model_path.is_absolute():
            candidate = (model_dir / base_model_path_str)
            if candidate.exists():
                base_model_path = candidate
            else:
                base_model_path = model_dir.parent / base_model_path_str
        if not base_model_path.exists():
            raise RuntimeError(f'Base model not found at: {base_model_path}. LoRA adapters require the base model to be available.')
        base_model_kwargs = {'pretrained_model_name_or_path': str(base_model_path), 'local_files_only': True, 'torch_dtype': model_kwargs.get('torch_dtype', 'auto'), 'low_cpu_mem_usage': True}
        if trust_remote:
            base_model_kwargs['trust_remote_code'] = True
        try:
            base_model = AutoModelForCausalLM.from_pretrained(**base_model_kwargs)
        except Exception as exc:
            raise RuntimeError(f'Failed to load base model from {base_model_path}: {exc}') from exc
        try:
            lora_model = PeftModel.from_pretrained(base_model, str(model_dir), local_files_only=True)
            return lora_model
        except Exception as exc:
            raise RuntimeError(f'Failed to load LoRA adapters: {exc}') from exc

    def _load_model(self, factory, kwargs: Dict[str, Any]):
        import torch as torch_module
        try:
            return factory.from_pretrained(**kwargs)
        except ValueError as exc:
            message = str(exc)
            if 'Unrecognized configuration class' in message or 'trust_remote_code' in message or 'requires you to execute' in message:
                retry_kwargs = dict(kwargs)
                retry_kwargs['trust_remote_code'] = True
                return factory.from_pretrained(**retry_kwargs)
            raise
        except RuntimeError as exc:
            if not self._should_retry_on_oom(exc):
                raise
            self._device = 'cpu'
            self._empty_device_caches()
            retry_kwargs = dict(kwargs)
            retry_kwargs['torch_dtype'] = torch_module.float32
            retry_kwargs['low_cpu_mem_usage'] = True
            return factory.from_pretrained(**retry_kwargs)

    def _load_tokenizer(self, factory, kwargs: Dict[str, Any], model_dir: Path):
        try:
            return factory.from_pretrained(**kwargs)
        except ValueError as exc:
            message = str(exc)
            if 'Unrecognized configuration class' in message or 'trust_remote_code' in message or 'requires you to execute' in message:
                retry_kwargs = dict(kwargs)
                retry_kwargs['trust_remote_code'] = True
                try:
                    return factory.from_pretrained(**retry_kwargs)
                except ValueError:
                    return self._load_tokenizer_override(factory, retry_kwargs, model_dir, str(exc))
            return self._load_tokenizer_override(factory, kwargs, model_dir, message)
        except OSError as exc:
            return self._load_tokenizer_override(factory, kwargs, model_dir, str(exc))

    def _load_tokenizer_override(self, factory, kwargs: Dict[str, Any], model_dir: Path, original_message: str):
        override_path = self._resolve_tokenizer_override(model_dir)
        if not override_path:
            raise RuntimeError(f"Tokenizer could not be loaded for {model_dir.name}. {original_message}. Add a LLaMA-compatible tokenizer by either (1) dropping tokenizer.json/tokenizer.model/tokenizer_config.json/special_tokens_map.json into a 'tokenizer' subfolder inside the model directory, or (2) creating tokenizer_source.txt next to the model with a relative or absolute path to an existing tokenizer (e.g. '../Llama-3.2-3B-Instruct').")
        override_kwargs = dict(kwargs)
        override_kwargs['pretrained_model_name_or_path'] = override_path
        override_kwargs.setdefault('local_files_only', True)
        try:
            return factory.from_pretrained(**override_kwargs)
        except Exception as exc:
            raise RuntimeError(f'Tokenizer override at {override_path} failed: {exc}. Ensure the tokenizer directory contains tokenizer.json/tokenizer.model.') from exc

    def _resolve_tokenizer_override(self, model_dir: Path) -> Optional[str]:
        explicit = model_dir / 'tokenizer_source.txt'
        if explicit.exists():
            for raw_line in explicit.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                candidate = self._expand_tokenizer_hint(model_dir, line)
                if candidate and self._looks_like_tokenizer_dir(candidate):
                    return str(candidate)
        fallback_dir = model_dir / 'tokenizer'
        if fallback_dir.exists() and self._looks_like_tokenizer_dir(fallback_dir):
            return str(fallback_dir)
        sibling = model_dir.with_name(f'{model_dir.name}-tokenizer')
        if sibling.exists() and self._looks_like_tokenizer_dir(sibling):
            return str(sibling)
        return None

    def _expand_tokenizer_hint(self, model_dir: Path, hint: str) -> Optional[Path]:
        guess = Path(hint)
        search_order = []
        if guess.is_absolute():
            search_order.append(guess)
        else:
            search_order.extend([(model_dir / guess).resolve(), (Path.cwd() / guess).resolve(), (model_dir.parent / guess).resolve()])
        for path in search_order:
            if path.exists():
                return path
        return None

    def _looks_like_tokenizer_dir(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        expected = {'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}
        entries = {entry.name for entry in path.iterdir()}
        return bool(expected & entries)

    def _should_retry_on_oom(self, exc: Exception) -> bool:
        import torch as torch_module
        if hasattr(torch_module, 'cuda') and isinstance(exc, torch_module.cuda.OutOfMemoryError):
            return True
        text = str(exc).lower()
        if 'out of memory' in text or 'mps backend' in text or 'unable to allocate' in text:
            return self._device != 'cpu'
        return False

    def _empty_device_caches(self) -> None:
        try:
            import torch as torch_module
            if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            if hasattr(torch_module, 'mps') and hasattr(torch_module.mps, 'empty_cache'):
                torch_module.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _should_trust_remote_code(self, model_dir: Path) -> bool:
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            return False
        try:
            config_data = json.loads(config_path.read_text())
        except Exception:
            return False
        auto_map = config_data.get('auto_map')
        if not auto_map:
            return False
        if isinstance(auto_map, dict):
            for value in auto_map.values():
                if self._contains_remote_reference(value, model_dir):
                    return True
        return False

    def _contains_remote_reference(self, value: Any, model_dir: Path) -> bool:
        if isinstance(value, str):
            if value.startswith('transformers_modules.'):
                return True
            module_name = value.split('.')[0]
            candidate = model_dir / f"{module_name.replace('.', '/')}.py"
            if candidate.exists():
                return True
            candidate = model_dir / 'src' / f"{module_name.replace('.', '/')}.py"
            if candidate.exists():
                return True
            return False
        if isinstance(value, list):
            return any((self._contains_remote_reference(item, model_dir) for item in value))
        return False

    @property
    def name(self) -> str:
        return 'transformers'

    def runtime_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {'device': self._device}
        try:
            param = next(self._model.parameters())
            info['dtype'] = str(param.dtype)
        except Exception:
            pass
        context_limit = getattr(self._model.config, 'max_position_embeddings', None)
        if isinstance(context_limit, int):
            info['max_position_embeddings'] = context_limit
        tokenizer_limit = getattr(self._tokenizer, 'model_max_length', None)
        if isinstance(tokenizer_limit, int):
            info['tokenizer_max_length'] = tokenizer_limit
        return info

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        import torch as torch_module
        add_special = bool(getattr(self._tokenizer, 'add_bos_token', False))
        encoded = self._tokenizer(prompt_text, return_tensors='pt', add_special_tokens=add_special)
        if not add_special and getattr(self._tokenizer, 'bos_token_id', None) is not None and (encoded['input_ids'].shape[1] > 0):
            bos_id = self._tokenizer.bos_token_id
            if bos_id is not None and encoded['input_ids'][0, 0].item() != bos_id:
                bos_tensor = torch_module.tensor([[bos_id]], device=encoded['input_ids'].device)
                encoded['input_ids'] = torch_module.cat([bos_tensor, encoded['input_ids']], dim=1)
                if 'attention_mask' in encoded:
                    encoded['attention_mask'] = torch_module.cat([torch_module.ones((1, 1), device=encoded['attention_mask'].device, dtype=encoded['attention_mask'].dtype), encoded['attention_mask']], dim=1)
        if self._device != 'cpu':
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        max_ctx = getattr(self._model.config, 'max_position_embeddings', getattr(self._tokenizer, 'model_max_length', 2048))
        input_len = int(encoded['input_ids'].shape[1])
        desired_new = int(cfg.max_tokens)
        if input_len + desired_new > max_ctx:
            keep = max(1, max_ctx - desired_new)
            encoded['input_ids'] = encoded['input_ids'][:, -keep:]
            if 'attention_mask' in encoded:
                encoded['attention_mask'] = encoded['attention_mask'][:, -keep:]
            input_len = int(encoded['input_ids'].shape[1])
        new_tokens = max(1, min(desired_new, max_ctx - input_len))
        do_sample = cfg.temperature > 0.0
        attention = encoded.get('attention_mask')
        prompt_ids = encoded['input_ids']
        eos_token_id = getattr(self._tokenizer, 'eos_token_id', None)
        model_vocab = int(getattr(self._model.config, 'vocab_size', 0) or 0)
        if model_vocab:
            prompt_ids = prompt_ids.clamp(min=0, max=model_vocab - 1)
        generation_ids = prompt_ids.clone().cpu()
        temperature = float(cfg.temperature) if do_sample else 1.0
        temperature = max(temperature, 0.0001)
        total_ids = prompt_ids
        attn_ids = attention
        with torch_module.no_grad():
            for _ in range(new_tokens):
                if model_vocab:
                    total_ids = total_ids.clamp(min=0, max=model_vocab - 1)
                outputs = self._model(input_ids=total_ids, attention_mask=attn_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
                vocab_size = getattr(self._model.config, 'vocab_size', logits.shape[-1])
                if do_sample:
                    probs = torch_module.softmax(logits / temperature, dim=-1)
                    next_token = torch_module.multinomial(probs, num_samples=1)
                else:
                    next_token = torch_module.argmax(logits, dim=-1, keepdim=True)
                token_value = int(next_token.item())
                if token_value < 0 or token_value >= vocab_size:
                    token_value = max(0, min(vocab_size - 1, token_value))
                    next_token = torch_module.tensor([[token_value]], device=total_ids.device, dtype=total_ids.dtype)
                total_ids = torch_module.cat([total_ids, next_token], dim=1)
                if attn_ids is not None:
                    ones = torch_module.ones((attn_ids.shape[0], 1), dtype=attn_ids.dtype, device=attn_ids.device)
                    attn_ids = torch_module.cat([attn_ids, ones], dim=1)
                generation_ids = torch_module.cat([generation_ids, next_token.cpu()], dim=1)
                if eos_token_id is not None and token_value == eos_token_id:
                    break
                if total_ids.shape[1] > max_ctx:
                    shift = total_ids.shape[1] - max_ctx
                    total_ids = total_ids[:, -max_ctx:]
                    if attn_ids is not None:
                        attn_ids = attn_ids[:, -max_ctx:]
        text = self._tokenizer.decode(generation_ids[0], skip_special_tokens=True)
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text):]
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        try:
            import torch as torch_module
            del self._model
            del self._tokenizer
            if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            if hasattr(torch_module, 'mps') and hasattr(torch_module.mps, 'empty_cache'):
                torch_module.mps.empty_cache()
        except Exception:
            pass

    def format_chat(self, messages: List[dict]) -> str:
        template = getattr(self._tokenizer, 'chat_template', None)
        if not template:
            return ''
        enable_thinking: Optional[bool] = None
        for item in messages:
            content = str(item.get('content', ''))
            if '/no_think' in content:
                enable_thinking = False
            elif '/think' in content:
                enable_thinking = True
        kwargs = dict(messages=messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is None:
            kwargs['enable_thinking'] = False
        else:
            kwargs['enable_thinking'] = enable_thinking
        try:
            return self._tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop('enable_thinking', None)
            try:
                return self._tokenizer.apply_chat_template(**kwargs)
            except Exception:
                return ''
        except Exception:
            return ''

class AutoGPTQBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for AutoGPTQ backends. Please install PyTorch for Jetson.')
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('auto-gptq is required but not installed. On Jetson, you may need to build it from source. See: https://github.com/PanQiWei/AutoGPTQ#installation') from exc
        except Exception as exc:
            raise RuntimeError(f'Failed to import auto-gptq: {exc}') from exc
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
        self._device = pick_device(device_pref)
        if self._device == 'cpu':
            device_map = None
        elif self._device == 'cuda':
            device_map = 'auto'
        else:
            device_map = 'auto' if self._device != 'cpu' else None
        try:
            self._model = AutoGPTQForCausalLM.from_quantized(str(model_dir), device_map=device_map, dtype='auto', use_safetensors=True, local_files_only=True)
        except Exception as exc:
            if device_map == 'auto' and self._device == 'cuda':
                try:
                    self._model = AutoGPTQForCausalLM.from_quantized(str(model_dir), device_map={'': 0}, dtype='auto', use_safetensors=True, local_files_only=True)
                except Exception as exc2:
                    raise RuntimeError(f'Failed to load AutoGPTQ model on CUDA: {exc}. Fallback also failed: {exc2}. Try setting device preference to CPU, or check that auto-gptq supports your Jetson configuration.') from exc2
            else:
                raise RuntimeError(f'Failed to load AutoGPTQ model: {exc}') from exc
        if self._device == 'mps':
            self._model.to('mps')
        elif self._device == 'cuda' and device_map is None:
            self._model.to('cuda')

    @property
    def name(self) -> str:
        return 'auto_gptq'

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        encoded = self._tokenizer(prompt_text, return_tensors='pt')
        if self._device != 'cpu':
            import torch
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        try:
            generated = self._model.generate(**encoded, max_new_tokens=cfg.max_tokens, do_sample=cfg.temperature > 0, temperature=cfg.temperature, pad_token_id=self._tokenizer.eos_token_id, eos_token_id=self._tokenizer.eos_token_id)
        except Exception as exc:
            raise RuntimeError(f'AutoGPTQ generation failed: {exc}') from exc
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text):]
        if cfg.stop:
            for stop_token in cfg.stop:
                if stop_token in text:
                    text = text.split(stop_token, 1)[0]
        return text.strip()

    def unload(self) -> None:
        try:
            import torch as torch_module
            del self._model
            del self._tokenizer
            if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            if hasattr(torch_module, 'mps') and hasattr(torch_module.mps, 'empty_cache'):
                torch_module.mps.empty_cache()
        except Exception:
            pass

class MLXBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        try:
            from pathlib import Path
            if Path('/etc/nv_tegra_release').exists():
                raise RuntimeError('MLX models are not supported on NVIDIA Jetson devices. MLX is Apple Silicon (M1/M2/M3) only. Please use GGUF models (via llama.cpp) or Transformers models instead.')
        except Exception:
            pass
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise RuntimeError("mlx_lm is required for MLX quantized models. MLX is only available on Apple Silicon (M1/M2/M3) devices. If you're on Jetson, use GGUF or Transformers models instead.")
        except Exception as exc:
            raise RuntimeError(f'Failed to load MLX model: {exc}') from exc
        def _try_load(**kwargs):
            return mlx_load(str(model_dir), **kwargs)

        last_exc: Optional[Exception] = None
        try:
            self._model, self._tokenizer = _try_load()
            return
        except Exception as exc:
            last_exc = exc

        # Force fast tokenizer with explicit file to avoid slow/sentencepiece path.
        tok_file = None
        for candidate in (model_dir / 'tokenizer.json', model_dir / 'tokenizer' / 'tokenizer.json'):
            if candidate.exists():
                tok_file = candidate
                break
        fast_kwargs: Dict[str, Any] = {'tokenizer_config': {'use_fast': True}}
        if tok_file:
            fast_kwargs['tokenizer_config']['tokenizer_file'] = str(tok_file)
        try:
            self._model, self._tokenizer = _try_load(**fast_kwargs)
            return
        except Exception as exc:
            last_exc = exc

        # As a last resort, try the slow tokenizer path only if sentencepiece is present.
        has_sentencepiece = True
        try:
            import sentencepiece  # type: ignore
        except Exception:
            has_sentencepiece = False
        if has_sentencepiece:
            try:
                self._model, self._tokenizer = _try_load(tokenizer_config={'use_fast': False})
                return
            except Exception as exc:
                last_exc = exc

        note = ''
        if not has_sentencepiece:
            note = ' (Install sentencepiece: pip install sentencepiece)'
        raise RuntimeError(f'Failed to load MLX model: {last_exc}{note}') from last_exc

    @property
    def name(self) -> str:
        return 'mlx_lm'

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        try:
            from mlx_lm import generate as mlx_generate
        except Exception as exc:
            raise RuntimeError(f'mlx_lm.generate unavailable: {exc}') from exc
        max_tokens = int(max(1, cfg.max_tokens))
        temperature = float(cfg.temperature if cfg.temperature > 0 else 0.01)
        kwargs = dict(prompt=prompt_text, max_tokens=max_tokens, temperature=temperature, verbose=False)
        try:
            result = mlx_generate(self._model, self._tokenizer, **kwargs)
        except TypeError:
            kwargs.pop('temperature', None)
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
        tokenizer = getattr(self, '_tokenizer', None)
        if tokenizer is None:
            return ''
        template = getattr(tokenizer, 'chat_template', None)
        if not template:
            return ''
        enable_thinking: Optional[bool] = None
        for item in messages:
            content = str(item.get('content', ''))
            if '/no_think' in content:
                enable_thinking = False
            elif '/think' in content:
                enable_thinking = True
        kwargs = dict(messages=messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is None:
            kwargs['enable_thinking'] = False
        else:
            kwargs['enable_thinking'] = enable_thinking
        try:
            return tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop('enable_thinking', None)
            try:
                return tokenizer.apply_chat_template(**kwargs)
            except Exception:
                return ''
        except Exception:
            return ''

class PhiVisionBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for vision models. Please install PyTorch for Jetson.')
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        except Exception:
            raise RuntimeError('transformers and torch required for vision models')
        import torch as torch_module
        self._device = pick_device(device_pref)
        torch_dtype = torch_module.float16 if self._device in {'cuda', 'mps'} else torch_module.float32
        self._processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        img_proc = getattr(self._processor, 'image_processor', None)
        if img_proc is not None and hasattr(img_proc, 'num_crops'):
            try:
                img_proc.num_crops = 1
            except Exception:
                try:
                    setattr(img_proc, 'num_crops', 1)
                except Exception:
                    pass
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        for attr in ('use_flash_attn', 'use_flash_attention', 'use_flash_attention_2', 'enable_flash_attn_2', 'enable_flash_attention_2'):
            if hasattr(config, attr):
                setattr(config, attr, False)
        if hasattr(config, 'attn_implementation'):
            config.attn_implementation = 'eager'
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        self._model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, dtype=torch_dtype, trust_remote_code=True, config=config, attn_implementation='eager')
        if self._device != 'cpu':
            self._model.to(self._device)
        self._model.eval()
        self._apply_template = getattr(self._processor, 'apply_chat_template', None)

    def _build_prompt(self, question: str, *, include_image: bool) -> str:
        """
        Build a chat prompt. Prefer the processor's chat template so images are correctly
        injected; fall back to a manual prompt with <|image_1|> tokens if that fails.
        """
        if callable(self._apply_template):
            # Structured messages allow processors to wire image embeddings correctly.
            if include_image:
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': question},
                        ],
                    }
                ]
            else:
                messages = [{'role': 'user', 'content': question}]
            try:
                prompt = self._apply_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(prompt, str) and prompt.endswith('<|endoftext|>'):
                    prompt = prompt.rsplit('<|endoftext|>', 1)[0]
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
            except Exception:
                pass
        user_prefix = '<|user|>\n'
        if include_image:
            user_prefix += '<|image_1|>\n'
        return f'{user_prefix}{question}<|end|>\n<|assistant|>\n'

    @property
    def name(self) -> str:
        return 'phi_vision'

    def generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        prompt = self._build_prompt(prompt_text, include_image=False)
        inputs = self._processor(text=prompt, return_tensors='pt')
        if self._device != 'cpu':
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        import torch as torch_module
        max_new = max(1, min(int(cfg.max_tokens), 160))
        with torch_module.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None, eos_token_id=self._processor.tokenizer.eos_token_id, pad_token_id=self._processor.tokenizer.pad_token_id, use_cache=False)
        new_tokens = generated[:, inputs['input_ids'].shape[1]:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return text[0].strip()

    def analyze_image(self, image_path: Path, prompt: str, cfg: GenerationConfig) -> str:
        image = Image.open(image_path).convert('RGB')
        max_side = max(image.size)
        if max_side > 1024:
            scale = 1024 / max_side
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.BICUBIC)
        chat_prompt = self._build_prompt(prompt, include_image=True)
        inputs = self._processor(text=chat_prompt, images=[image], return_tensors='pt')
        if self._device != 'cpu':
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        max_new = max(64, min(int(cfg.max_tokens), 160))
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=min(max_new, 256), do_sample=False, temperature=None, eos_token_id=self._processor.tokenizer.eos_token_id, pad_token_id=self._processor.tokenizer.pad_token_id, use_cache=False)
        new_tokens = generated[:, inputs['input_ids'].shape[1]:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return text[0].strip()

    def unload(self) -> None:
        try:
            import torch
            del self._model
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception:
            pass
        self._processor = None

class DiffusersT2IBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for diffusers backends. Please install PyTorch for Jetson.')
        try:
            from diffusers import AutoPipelineForText2Image
        except Exception:
            raise RuntimeError('diffusers, transformers and torch required')
        self._device = pick_device(device_pref)
        from diffusers import AutoPipelineForText2Image
        self._pipeline = AutoPipelineForText2Image.from_pretrained(str(model_dir), local_files_only=True, dtype='auto', use_safetensors=True)
        try:
            self._pipeline.safety_checker = None
        except Exception:
            pass
        self._pipeline = self._pipeline.to(self._device)

    @property
    def name(self) -> str:
        return 'diffusers_t2i'

    def _align(self, value: int) -> int:
        return max(8, int(value // 8 * 8))

    def generate_image(self, prompt: str, steps: int=4, guidance: float=0.0, width: int=512, height: int=512, seed: Optional[int]=None, outdir: Path=Path('outputs')) -> str:
        import torch as torch_module
        outdir.mkdir(parents=True, exist_ok=True)
        width = self._align(width)
        height = self._align(height)
        steps = max(1, min(steps, 50))
        generator = torch_module.Generator(device=self._device)
        if seed is None:
            seed = int(time.time()) & 4294967295
        generator = generator.manual_seed(int(seed))
        try:
            result = self._pipeline(prompt, num_inference_steps=steps, guidance_scale=float(guidance), width=width, height=height, generator=generator)
            image = result.images[0]
        except Exception as exc:
            if 'embedding images is forbidden' in str(exc).lower():
                raise RuntimeError('This diffusion model only supports pure text prompts (no init image).') from exc
            raise RuntimeError(f'Diffusers generation failed: {exc}') from exc
        path = outdir / f'img_{seed}.png'
        image.save(path)
        return str(path)

    def unload(self) -> None:
        try:
            import torch as torch_module
            self._pipeline = None
            if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            if hasattr(torch_module, 'mps') and hasattr(torch_module.mps, 'empty_cache'):
                torch_module.mps.empty_cache()
        except Exception:
            pass

class OCRBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for OCR backends. Please install PyTorch for Jetson.')
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError('transformers (vision) required for OCR')
        device_str = pick_device(device_pref)
        if device_str == 'cpu':
            pipeline_device = -1
        elif device_str == 'cuda':
            pipeline_device = 0
        elif device_str == 'mps':
            pipeline_device = 'mps'
        else:
            pipeline_device = -1
        self._device = device_str
        self._pipeline = pipeline('image-to-text', model=str(model_dir), tokenizer=None, feature_extractor=None, device=pipeline_device, local_files_only=True)

    @property
    def name(self) -> str:
        return 'ocr_trocr'

    def run(self, image_path: str) -> str:
        try:
            outputs = self._pipeline(image_path)
            if isinstance(outputs, list) and outputs and ('generated_text' in outputs[0]):
                return outputs[0]['generated_text']
            return str(outputs)
        except Exception as exc:
            raise RuntimeError(f'OCR failed: {exc}') from exc

    def unload(self) -> None:
        try:
            import torch
            if self._device == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipeline = None

class ASRBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for ASR backends. Please install PyTorch for Jetson.')
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError('transformers (audio) required for ASR')
        device_str = pick_device(device_pref)
        if device_str == 'cpu':
            pipeline_device = -1
        elif device_str == 'cuda':
            pipeline_device = 0
        elif device_str == 'mps':
            pipeline_device = 'mps'
        else:
            pipeline_device = -1
        self._device = device_str
        self._pipeline = pipeline('automatic-speech-recognition', model=str(model_dir), device=pipeline_device, local_files_only=True)

    @property
    def name(self) -> str:
        return 'asr_whisper'

    def run(self, audio_path: str) -> str:
        try:
            output = self._pipeline(audio_path)
            if isinstance(output, dict) and 'text' in output:
                return output['text']
            return str(output)
        except Exception as exc:
            raise RuntimeError(f'ASR failed: {exc}') from exc

    def unload(self) -> None:
        try:
            import torch
            if self._device == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipeline = None

class TTSBackend(BaseBackend):

    def __init__(self, model_dir: Path, device_pref: str):
        if torch is None:
            raise RuntimeError('PyTorch (torch) is required for TTS backends. Please install PyTorch for Jetson.')
        try:
            from transformers import pipeline
        except Exception:
            raise RuntimeError('transformers (TTS) required for text-to-speech')
        device_str = pick_device(device_pref)
        if device_str == 'cpu':
            pipeline_device = -1
        elif device_str == 'cuda':
            pipeline_device = 0
        elif device_str == 'mps':
            pipeline_device = 'mps'
        else:
            pipeline_device = -1
        self._device = device_str
        self._pipeline = pipeline('text-to-speech', model=str(model_dir), device=pipeline_device, local_files_only=True)

    @property
    def name(self) -> str:
        return 'tts_transformers'

    def run(self, text: str, outdir: Path) -> str:
        try:
            audio = self._pipeline(text)
            samples = audio['audio']
            sample_rate = int(audio.get('sampling_rate', 22050))
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f'tts_{int(time.time())}.wav'
            sf.write(str(path), samples, sample_rate)
            return str(path)
        except Exception as exc:
            raise RuntimeError(f'TTS failed: {exc}') from exc

    def unload(self) -> None:
        try:
            import torch
            if self._device == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipeline = None
