from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


APP_NAME = "ACRE"
_PORTABLE_MARKER = ".acre_portable"
_IGNORE_NAMES = {".DS_Store", ".gitkeep"}


def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def legacy_root() -> Path:
    return app_root()


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_user_path(raw: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _platform_data_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_NAME.lower()
    return Path.home() / ".local" / "share" / APP_NAME.lower()


def data_root() -> Path:
    override = os.environ.get("ACRE_DATA_DIR")
    if override:
        return _resolve_user_path(override)

    if _truthy(os.environ.get("ACRE_PORTABLE")):
        return legacy_root()

    try:
        marker = legacy_root() / _PORTABLE_MARKER
        if marker.exists():
            return legacy_root()
    except Exception:
        pass

    return _platform_data_root()


def user_config_dir() -> Path:
    return data_root() / "config"


def user_config_path() -> Path:
    return user_config_dir() / "settings.json"


def legacy_config_path() -> Path:
    return legacy_root() / "config" / "settings.json"


def history_dir() -> Path:
    return data_root() / "history"


def outputs_dir() -> Path:
    return data_root() / "outputs"


def _iter_non_ignored(entries: Iterable[Path]) -> Iterable[Path]:
    for entry in entries:
        if entry.name in _IGNORE_NAMES:
            continue
        yield entry


def _dir_has_user_content(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
        return any(True for _ in _iter_non_ignored(path.iterdir()))
    except Exception:
        return False


def _load_settings_raw() -> dict:
    for candidate in (user_config_path(), legacy_config_path()):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _models_dir_from_settings(settings: dict) -> Optional[str]:
    paths_section = settings.get("paths")
    if isinstance(paths_section, dict):
        candidate = paths_section.get("models_dir")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    legacy_key = settings.get("models_dir")
    if isinstance(legacy_key, str) and legacy_key.strip():
        return legacy_key.strip()
    return None


def models_dir() -> Path:
    override = os.environ.get("ACRE_MODELS_DIR")
    if override:
        return _resolve_user_path(override)

    settings_override = _models_dir_from_settings(_load_settings_raw())
    if settings_override:
        return _resolve_user_path(settings_override)

    preferred = data_root() / "models"
    legacy = legacy_root() / "models"

    if _dir_has_user_content(preferred):
        return preferred
    if _dir_has_user_content(legacy):
        return legacy
    return preferred


def user_history_dir(username: Optional[str]) -> Path:
    base = history_dir()
    return base / username if username else base


def user_outputs_dir(username: Optional[str]) -> Path:
    base = outputs_dir()
    return base / username if username else base


def ensure_user_data_dirs() -> None:
    for path in (user_config_dir(), history_dir(), outputs_dir(), data_root() / "models"):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _copy_missing_files(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        try:
            rel = item.relative_to(src)
        except Exception:
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil

            shutil.copy2(item, target)
        except Exception:
            pass


def migrate_legacy_user_data(*, force: bool = False) -> None:
    """
    Best-effort migration from the legacy repo-local layout into the user data dir.

    - Copies `config/settings.json` if the new config does not exist.
    - Copies missing files from legacy `history/` and `outputs/` into the new location.
    - Does not copy models by default (models can be very large); `models_dir()` will
      keep using the legacy models folder until the user populates the new one.
    """
    src_root = legacy_root()
    dst_root = data_root()
    if src_root == dst_root:
        return
    marker = dst_root / ".acre_migration_complete"
    if marker.exists() and not force:
        return

    try:
        ensure_user_data_dirs()
    except Exception:
        return

    # Settings
    legacy_settings = legacy_config_path()
    new_settings = user_config_path()
    if legacy_settings.exists() and not new_settings.exists():
        try:
            import shutil

            new_settings.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_settings, new_settings)
        except Exception:
            pass

    # Histories + outputs (copy missing files only)
    _copy_missing_files(src_root / "history", history_dir())
    _copy_missing_files(src_root / "outputs", outputs_dir())

    try:
        dst_root.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass


__all__ = [
    "APP_NAME",
    "app_root",
    "legacy_root",
    "data_root",
    "user_config_dir",
    "user_config_path",
    "legacy_config_path",
    "models_dir",
    "history_dir",
    "outputs_dir",
    "user_history_dir",
    "user_outputs_dir",
    "ensure_user_data_dirs",
    "migrate_legacy_user_data",
]
