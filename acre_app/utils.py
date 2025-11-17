import os
import subprocess
import sys
from pathlib import Path


def to_int(val, default):
    try:
        text = str(val).strip()
        return int(text) if text not in ("", None) else default
    except Exception:
        return default


def to_float(val, default):
    try:
        text = str(val).strip()
        return float(text) if text not in ("", None) else default
    except Exception:
        return default


def open_path(path: Path) -> None:
    try:
        resolved = path.resolve()
        if not resolved.exists():
            return
        
        if sys.platform == "darwin":
            subprocess.run(["open", str(resolved)], check=False)
        elif os.name == "nt":
            os.startfile(str(resolved))
        else:
            subprocess.run(["xdg-open", str(resolved)], check=False)
    except Exception:
        pass
