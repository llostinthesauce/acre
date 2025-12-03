from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def is_jetson() -> bool:
    try:
        if Path("/etc/nv_tegra_release").exists():
            return True
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64") and "jetson" in platform.platform().lower():
            return True
        uname = platform.uname()
        if "jetson" in uname.system.lower() or "jetson" in uname.release.lower():
            return True
    except Exception:
        pass
    return False


@lru_cache(maxsize=1)
def is_arm64_linux() -> bool:
    try:
        machine = platform.machine().lower()
        system = platform.system().lower()
        if machine in ("aarch64", "arm64") and system == "linux":
            return True
    except Exception:
        pass
    return False
