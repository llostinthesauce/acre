#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_utils import is_jetson

JETPACK_RELEASE_FILE = Path("/etc/nv_tegra_release")
WHEEL_RECOMMENDATIONS = {
    "5.1": {
        "torch": "torch==2.1.1+nv",
        "torchvision": "torchvision==0.15.2+nv",
        "torchaudio": "torchaudio==2.1.2+nv",
        "index_url": "https://developer.download.nvidia.com/compute/redist/jp/v12.0/pytorch/torch_stable.html",
    },
}


def _read_jetpack_version() -> Optional[str]:
    if not JETPACK_RELEASE_FILE.exists():
        return None
    try:
        lines = JETPACK_RELEASE_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for line in lines:
        match = re.search(r"JetPack\s*([\d\.]+)", line)
        if match:
            return match.group(1)
    for line in lines:
        match = re.search(r"release\s+(\d+\.\d+)", line, re.IGNORECASE)
        if match:
            release_num = float(match.group(1))
            if release_num >= 35:
                return "5.1"
            if release_num >= 34:
                return "4.6"
    return None


def _format_command(info: dict[str, str]) -> str:
    parts = [
        info["torch"],
        info["torchvision"],
        info["torchaudio"],
        "-f",
        info["index_url"],
    ]
    return "pip install " + " ".join(parts)


def main() -> None:
    if not is_jetson():
        print("This helper is for NVIDIA Jetson platforms.")
        print("Run from a Jetson device after installing the required dependencies.")
        return

    version = _read_jetpack_version()
    print(f"Detected JetPack version: {version or 'unknown'}")
    recommendation = None
    if version:
        for key in WHEEL_RECOMMENDATIONS:
            if version.startswith(key):
                recommendation = WHEEL_RECOMMENDATIONS[key]
                break
    if not recommendation and version and version.startswith("5.1"):
        recommendation = WHEEL_RECOMMENDATIONS["5.1"]

    if recommendation:
        print("Suggested PyTorch install command (JetPack-compatible wheel):")
        print(_format_command(recommendation))
    else:
        print("Please visit https://developer.download.nvidia.com/compute/redist/jp/ and download the wheel")
        print("that matches your JetPack/CUDA release before installing.")

    print("\nAfter installing PyTorch, run:")
    print("  bash install_cusparselt.sh")
    print("\nSee docs/jetson_training.md for additional training guidance.")


if __name__ == "__main__":
    main()
