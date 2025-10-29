import os
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
    resolved = path.resolve()
    if sys.platform == "darwin":
        os.system(f'open "{resolved}"')
        return
    if os.name == "nt":
        os.startfile(str(resolved))
        return
    os.system(f'xdg-open "{resolved}"')
