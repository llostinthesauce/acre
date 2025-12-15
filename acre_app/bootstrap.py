from __future__ import annotations

import importlib.util
import os
import sys

from platform_utils import is_arm64_linux, is_jetson

from . import paths

APP_ROOT = paths.app_root()
VENDOR = APP_ROOT / "vendor"


def _need(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except Exception:
        return True


def _configure_vendor_path() -> None:
    if VENDOR.exists() and str(VENDOR) not in sys.path:
        # Keep vendor at the end so system/site-packages versions win (avoids stale vendored wheels).
        sys.path.append(str(VENDOR))


def setup_environment() -> None:
    # Ensure all writable user data lives in the OS data directory (or user override).
    # This is intentionally best-effort and should never prevent the app from starting.
    try:
        paths.ensure_user_data_dirs()
        paths.migrate_legacy_user_data()
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_ALLOW_CODE_DOWNLOAD", "1")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    _configure_vendor_path()

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

    # Offline-hardening: do not auto-install packages at runtime.
    # If core GUI deps are missing, fail fast with a clear message.
    required = [
        ("customtkinter", "customtkinter"),
        ("PIL", "pillow"),
        ("cryptography", "cryptography"),
        ("soundfile", "soundfile"),
    ]
    missing = [pkg for mod, pkg in required if _need(mod)]
    if missing:
        joined = ", ".join(missing)
        print(f"ERROR: Missing required Python dependencies: {joined}", file=sys.stderr)
        print("Install them with:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print("Offline builds should ship vendored wheels; see `notes/deps.md`.", file=sys.stderr)
        raise SystemExit(1)
