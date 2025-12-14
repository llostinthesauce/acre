import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from platform_utils import is_arm64_linux, is_jetson

from .constants import MODELS_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
VENDOR = BASE_DIR / "vendor"


def _runtime_installs_enabled() -> bool:
    flag = os.environ.get("ACRE_ENABLE_RUNTIME_INSTALLS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _need(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except Exception:
        return True


def _ensure(pkgs: list[tuple[str, str]]) -> None:
    missing = [p for p in pkgs if _need(p[1])]
    if not missing:
        return
    
    arm64_linux = is_arm64_linux()
    filtered_pkgs = []
    for pkg_name, mod_name in missing:
        if arm64_linux and (pkg_name.startswith("torch") or mod_name == "torch"):
            print(f"SKIPPING {pkg_name} on ARM64 Linux - PyPI wheels are not supported.")
            print("  Please install PyTorch from NVIDIA's Jetson wheels or build from source.")
            continue
        filtered_pkgs.append((pkg_name, mod_name))
    
    if not filtered_pkgs:
        return
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-t", str(VENDOR)]
            + [p[0] for p in filtered_pkgs]
        )
    except Exception as exc:
        names = ", ".join(p[0] for p in filtered_pkgs)
        error_msg = str(exc)
        if "torch" in names.lower() and ("linux" in error_msg.lower() or "arch" in error_msg.lower() or "wheel" in error_msg.lower()):
            print(f"ERROR: Failed to install {names}")
            print("  PyTorch from PyPI is not supported on ARM64 Linux systems.")
            print("  On Jetson devices, install PyTorch from NVIDIA's pre-built wheels.")
        else:
            print(f"Failed to install {names}: {exc}")


def _requires_diffusers() -> bool:
    if not MODELS_PATH.exists():
        return False
    for entry in MODELS_PATH.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "model_index.json").exists():
            return True
    return False


def _requires_mlx() -> bool:
    if is_jetson():
        return False
    if not MODELS_PATH.exists():
        return False
    for entry in MODELS_PATH.iterdir():
        if not entry.is_dir():
            continue
        if "mlx" in entry.name.lower():
            return True
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text())
        except Exception:
            continue
        quant = data.get("quantization_config") or {}
        if quant and "mlx" in str(quant).lower():
            return True
    return False


def _requires_transformers() -> bool:
    if not MODELS_PATH.exists():
        return False
    for entry in MODELS_PATH.iterdir():
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text())
        except Exception:
            continue
        if data.get("architectures") or data.get("model_type"):
            return True
    return False


def _requires_training() -> bool:
    return _requires_transformers()


def setup_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ALLOW_CODE_DOWNLOAD", "1")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    VENDOR.mkdir(exist_ok=True)
    if str(VENDOR) not in sys.path:
        # Keep vendor at the end so system/site-packages versions win (avoids stale vendored wheels).
        sys.path.append(str(VENDOR))

    # Guard against numpy>=2 removing legacy aliases (e.g., Inf) used by some deps.
    try:
        import numpy as _np
        if not hasattr(_np, "Inf"):
            _np.Inf = _np.inf  # type: ignore[attr-defined]
    except Exception:
        pass
    
    on_jetson = is_jetson()
    if on_jetson:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    arm64_linux = is_arm64_linux()
    if arm64_linux and not on_jetson:
        print("WARNING: Detected ARM64 Linux system. PyTorch must be installed manually.")
        print("  Standard PyPI PyTorch wheels are not available for ARM64 Linux.")
    
    def _install_if_enabled(packages: list[tuple[str, str]], reason: str) -> None:
        if not _runtime_installs_enabled():
            print(f"Skipping runtime install for {reason} (set ACRE_ENABLE_RUNTIME_INSTALLS=1 to enable).")
            return
        _ensure(packages)

    base_packages = [
        ("customtkinter>=5.2.0", "customtkinter"),
        ("pillow>=10.2.0", "PIL"),
        ("darkdetect>=0.8.0", "darkdetect"),
        ("llama-cpp-python>=0.3.16", "llama_cpp"),
        ("send2trash>=1.8.3", "send2trash"),
        ("protobuf>=3.20,<6", "google.protobuf"),
        ("soundfile>=0.12", "soundfile"),
        ("pymupdf>=1.24", "fitz"),
        ("cryptography>=42.0.0", "cryptography"),
    ]
    _install_if_enabled(base_packages, "UI/runtime base packages")
    
    if _requires_transformers():
        if not arm64_linux:
            transformer_packages = [
                ("torch>=2.3", "torch"),
                ("torchvision>=0.18", "torchvision"),
                ("transformers>=4.52.4", "transformers"),
            ]
            _install_if_enabled(transformer_packages, "transformers backends")
        else:
            if not _need("torch"):
                transformers_only = [
                    ("transformers>=4.52.4", "transformers"),
                ]
                _install_if_enabled(transformers_only, "transformers-only backends")
            else:
                print("WARNING: PyTorch not found on ARM64 Linux system.")
                if on_jetson:
                    print("  Please install PyTorch for Jetson from NVIDIA's wheels.")
                else:
                    print("  Please install PyTorch manually (PyPI wheels not available for ARM64 Linux).")
    
    if _requires_mlx():
        mlx_packages = [
            ("mlx-lm>=0.25.2", "mlx_lm"),
        ]
        _install_if_enabled(mlx_packages, "MLX backends")
    
    if _requires_training() and not arm64_linux:
        training_packages = [
            ("datasets>=2.20.0", "datasets"),
        ]
        _install_if_enabled(training_packages, "training")
    elif _requires_training() and arm64_linux:
        if not _need("torch"):
            training_packages = [
                ("datasets>=2.20.0", "datasets"),
            ]
            _install_if_enabled(training_packages, "training")
    
    if not _requires_diffusers():
        return
    
    if not arm64_linux:
        diffusion_packages = [
            ("torch>=2.3", "torch"),
            ("transformers>=4.52.4", "transformers"),
            ("diffusers>=0.25", "diffusers"),
            ("safetensors>=0.4", "safetensors"),
            ("huggingface_hub>=0.23", "huggingface_hub"),
            ("accelerate>=0.31", "accelerate"),
        ]
        _install_if_enabled(diffusion_packages, "diffusion/image backends")
    else:
        if not _need("torch"):
            diffusion_packages = [
                ("transformers>=4.52.4", "transformers"),
                ("diffusers>=0.25", "diffusers"),
                ("safetensors>=0.4", "safetensors"),
                ("huggingface_hub>=0.23", "huggingface_hub"),
                ("accelerate>=0.31", "accelerate"),
            ]
            _install_if_enabled(diffusion_packages, "diffusion/image backends")
        else:
            print("WARNING: PyTorch not found on ARM64 Linux system.")
            if on_jetson:
                print("  Please install PyTorch for Jetson from NVIDIA's wheels.")
            else:
                print("  Please install PyTorch manually (PyPI wheels not available for ARM64 Linux).")
