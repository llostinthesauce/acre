import base64
import json
import secrets
from hashlib import pbkdf2_hmac
from typing import Dict, List, Optional

from .constants import CONFIG_PATH


def load_settings() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(CONFIG_PATH)


def encode_b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def decode_b64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("utf-8"))


def ensure_users_bucket(settings: dict) -> Dict[str, dict]:
    return settings.setdefault("users", {})


def list_usernames(settings: dict) -> List[str]:
    return sorted(ensure_users_bucket(settings).keys())


def get_active_user(settings: dict) -> Optional[str]:
    user = settings.get("active_user")
    return user if isinstance(user, str) and user in ensure_users_bucket(settings) else None


def set_active_user(settings: dict, username: str) -> None:
    settings["active_user"] = username
    save_settings(settings)


def set_credentials(password: str, iterations: int = 200_000) -> dict:
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_bytes(16)
    digest = pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    enc_salt = secrets.token_bytes(16)
    return {
        "password_hash": encode_b64(digest),
        "salt": encode_b64(salt),
        "iterations": iterations,
        "enc_salt": encode_b64(enc_salt),
        "enc_iterations": iterations,
        "disclaimer_ack": False,
    }


def verify_password(username: str, password: str, settings: dict) -> bool:
    record = ensure_users_bucket(settings).get(username)
    if not record:
        return False
    try:
        digest = decode_b64(record["password_hash"])
        salt = decode_b64(record["salt"])
        iterations = int(record["iterations"])
        candidate = pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return candidate == digest
    except Exception:
        return False


def get_prefs() -> dict:
    settings = load_settings()
    prefs = settings.setdefault("prefs", {})
    return {
        "text_temperature": float(prefs.get("text_temperature", 0.7)),
        "text_max_tokens": int(prefs.get("text_max_tokens", 512)),
        "image_width": int(prefs.get("image_width", 512)),
        "image_height": int(prefs.get("image_height", 512)),
        "image_steps": int(prefs.get("image_steps", 4)),
        "image_guidance": float(prefs.get("image_guidance", 0.0)),
        "image_seed": (
            None
            if prefs.get("image_seed") in (None, "", "None")
            else int(prefs.get("image_seed"))
        ),
        "ui_scale": float(prefs.get("ui_scale", 1.15)),
        "device_preference": str(prefs.get("device_preference", "auto")).lower(),
        "history_enabled": bool(prefs.get("history_enabled", True)),
    }


def set_prefs(new_prefs: dict) -> None:
    settings = load_settings()
    settings.setdefault("prefs", {}).update(new_prefs)
    save_settings(settings)


def per_user_bucket(settings: dict, username: str) -> dict:
    container = settings.setdefault("per_user", {})
    return container.setdefault(username, {"model_aliases": {}})


def get_alias_map(settings: dict, username: str) -> Dict[str, str]:
    return per_user_bucket(settings, username).setdefault("model_aliases", {})


def set_alias(settings: dict, username: str, real_name: str, alias: Optional[str]) -> None:
    mapping = get_alias_map(settings, username)
    if alias and alias.strip():
        mapping[real_name] = alias.strip()
    else:
        mapping.pop(real_name, None)
    save_settings(settings)


def get_user_record(settings: dict, username: str) -> Optional[dict]:
    record = ensure_users_bucket(settings).get(username)
    return record if isinstance(record, dict) else None


def ensure_encryption_metadata(settings: dict, username: str, *, default_iterations: Optional[int] = None) -> Optional[dict]:
    record = get_user_record(settings, username)
    if record is None:
        return None
    changed = False
    if "enc_salt" not in record:
        record["enc_salt"] = encode_b64(secrets.token_bytes(16))
        changed = True
    if "enc_iterations" not in record:
        iterations = default_iterations or record.get("iterations") or 200_000
        try:
            iterations = int(iterations)
        except Exception:
            iterations = 200_000
        record["enc_iterations"] = iterations
        changed = True
    if "disclaimer_ack" not in record:
        record["disclaimer_ack"] = True
        changed = True
    if changed:
        save_settings(settings)
    return record


def set_disclaimer_ack(settings: dict, username: str, value: bool) -> None:
    record = ensure_encryption_metadata(settings, username)
    if record is None:
        return
    if record.get("disclaimer_ack") == bool(value):
        return
    record["disclaimer_ack"] = bool(value)
    save_settings(settings)
