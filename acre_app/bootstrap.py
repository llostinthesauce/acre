import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .constants import MODELS_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
VENDOR = BASE_DIR / "vendor"


def _is_jetson() -> bool:
    """Detect if running on NVIDIA Jetson platform."""
    try:
        # Check for Jetson-specific files
        if Path("/etc/nv_tegra_release").exists():
            return True
        # Check machine architecture
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64") and "jetson" in platform.platform().lower():
            return True
        # Check for NVIDIA Jetson in uname
        uname = platform.uname()
        if "jetson" in uname.system.lower() or "jetson" in uname.release.lower():
            return True
    except Exception:
        pass
    return False


def _is_arm64_linux() -> bool:
    """Detect if running on ARM64 Linux (Jetson or other ARM64 systems)."""
    try:
        machine = platform.machine().lower()
        system = platform.system().lower()
        # ARM64 Linux systems (including Jetson) can't use PyPI PyTorch
        if machine in ("aarch64", "arm64") and system == "linux":
            return True
    except Exception:
        pass
    return False


def _need(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except Exception:
        return True


def _ensure(pkgs: list[tuple[str, str]]) -> None:
    missing = [p for p in pkgs if _need(p[1])]
    if not missing:
        return
    
    # Block PyTorch installation on ARM64 Linux (Jetson or other ARM64 systems)
    is_arm64_linux = _is_arm64_linux()
    filtered_pkgs = []
    for pkg_name, mod_name in missing:
        # Skip torch and torchvision on ARM64 Linux - they need special builds
        if is_arm64_linux and (pkg_name.startswith("torch") or mod_name == "torch"):
            print(f"SKIPPING {pkg_name} on ARM64 Linux - PyPI wheels are not supported.")
            print("  Please install PyTorch from NVIDIA's Jetson wheels or build from source.")
            print("  See JETSON_SETUP.md for instructions.")
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
        # Provide helpful error message for PyTorch installation failures
        if "torch" in names.lower() and ("linux" in error_msg.lower() or "arch" in error_msg.lower() or "wheel" in error_msg.lower()):
            print(f"ERROR: Failed to install {names}")
            print("  PyTorch from PyPI is not supported on ARM64 Linux systems.")
            print("  On Jetson devices, install PyTorch from NVIDIA's pre-built wheels.")
            print("  See JETSON_SETUP.md for detailed instructions.")
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
    # MLX is Apple Silicon only, skip on Jetson
    if _is_jetson():
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


def setup_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ALLOW_CODE_DOWNLOAD", "1")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    VENDOR.mkdir(exist_ok=True)
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    
    is_jetson = _is_jetson()
    is_arm64_linux = _is_arm64_linux()
    
    # Warn if on ARM64 Linux but not detected as Jetson
    if is_arm64_linux and not is_jetson:
        print("WARNING: Detected ARM64 Linux system. PyTorch must be installed manually.")
        print("  Standard PyPI PyTorch wheels are not available for ARM64 Linux.")
        print("  See JETSON_SETUP.md for installation guidance.")
    
    base_packages = [
        ("customtkinter>=5.2.0", "customtkinter"),
        ("pillow>=10.2.0", "PIL"),
        ("darkdetect>=0.8.0", "darkdetect"),
        ("llama-cpp-python>=0.3.16", "llama_cpp"),
        ("send2trash>=1.8.3", "send2trash"),
        ("protobuf>=3.20,<6", "google.protobuf"),
        ("soundfile>=0.12", "soundfile"),
        ("pymupdf>=1.24", "fitz"),
    ]
    _ensure(base_packages)
    
    if _requires_transformers():
        # On ARM64 Linux (Jetson or other), PyTorch should be pre-installed
        # Don't install from PyPI as it won't work on ARM64
        if not is_arm64_linux:
            transformer_packages = [
                ("torch>=2.3", "torch"),
                ("torchvision>=0.18", "torchvision"),
                ("transformers>=4.52.4", "transformers"),
            ]
            _ensure(transformer_packages)
        else:
            # On ARM64 Linux, only install transformers if torch is already available
            if not _need("torch"):
                transformers_only = [
                    ("transformers>=4.52.4", "transformers"),
                ]
                _ensure(transformers_only)
            else:
                print("WARNING: PyTorch not found on ARM64 Linux system.")
                if is_jetson:
                    print("  Please install PyTorch for Jetson from NVIDIA's wheels.")
                else:
                    print("  Please install PyTorch manually (PyPI wheels not available for ARM64 Linux).")
                print("  See JETSON_SETUP.md for instructions.")
    
    if _requires_mlx():
        mlx_packages = [
            ("mlx-lm>=0.25.2", "mlx_lm"),
        ]
        _ensure(mlx_packages)
    
    if not _requires_diffusers():
        return
    
    # For diffusers, same logic - skip torch on ARM64 Linux
    if not is_arm64_linux:
        diffusion_packages = [
            ("torch>=2.3", "torch"),
            ("transformers>=4.52.4", "transformers"),
            ("diffusers>=0.25", "diffusers"),
            ("safetensors>=0.4", "safetensors"),
            ("huggingface_hub>=0.23", "huggingface_hub"),
            ("accelerate>=0.31", "accelerate"),
        ]
        _ensure(diffusion_packages)
    else:
        # On ARM64 Linux, install diffusers dependencies if torch is available
        if not _need("torch"):
            diffusion_packages = [
                ("transformers>=4.52.4", "transformers"),
                ("diffusers>=0.25", "diffusers"),
                ("safetensors>=0.4", "safetensors"),
                ("huggingface_hub>=0.23", "huggingface_hub"),
                ("accelerate>=0.31", "accelerate"),
            ]
            _ensure(diffusion_packages)
        else:
            print("WARNING: PyTorch not found on ARM64 Linux system.")
            if is_jetson:
                print("  Please install PyTorch for Jetson from NVIDIA's wheels.")
            else:
                print("  Please install PyTorch manually (PyPI wheels not available for ARM64 Linux).")
            print("  See JETSON_SETUP.md for instructions.")
