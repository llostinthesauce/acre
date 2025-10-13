import base64
import json
import os
import secrets
import hmac
import hashlib
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken


class SecurityManager:
    """
    Handles password storage/verification and derives encryption keys
    for protecting chat history on disk.
    """

    def __init__(self, config_dir: str = "config", iterations: int = 480_000):
        self.config_dir = config_dir
        self.iterations = iterations
        self.config_path = os.path.join(self.config_dir, "settings.json")
        os.makedirs(self.config_dir, exist_ok=True)

        self._password_hash: bytes = b""
        self._salt: bytes = b""
        self._key: bytes = b""
        self._load_config()

    # ------------------------------------------------------------------ Config IO
    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return

        iterations = payload.get("iterations")
        if isinstance(iterations, int) and iterations > 0:
            self.iterations = iterations

        self._password_hash = base64.urlsafe_b64decode(payload.get("password_hash", "")) if payload.get("password_hash") else b""
        self._salt = base64.urlsafe_b64decode(payload.get("salt", "")) if payload.get("salt") else b""

    def _save_config(self) -> None:
        payload = {
            "password_hash": base64.urlsafe_b64encode(self._password_hash).decode("utf-8"),
            "salt": base64.urlsafe_b64encode(self._salt).decode("utf-8"),
            "iterations": self.iterations,
        }
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------ Passwords
    def is_password_set(self) -> bool:
        return bool(self._password_hash and self._salt)

    def set_password(self, password: str) -> None:
        self._salt = secrets.token_bytes(16)
        self._password_hash = self._hash_password(password, self._salt)
        self._key = self._derive_key(password, self._salt)
        self._save_config()

    def verify_password(self, password: str) -> bool:
        if not self.is_password_set():
            return False
        candidate_hash = self._hash_password(password, self._salt)
        if hmac.compare_digest(candidate_hash, self._password_hash):
            self._key = self._derive_key(password, self._salt)
            return True
        return False

    def change_password(self, current_password: str, new_password: str) -> bool:
        if not self.verify_password(current_password):
            return False
        self.set_password(new_password)
        return True

    def _hash_password(self, password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
        )

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        raw_key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
            dklen=32,
        )
        return base64.urlsafe_b64encode(raw_key)

    # ------------------------------------------------------------------ Encryption
    def _ensure_key(self) -> bytes:
        if not self._key:
            raise RuntimeError("Encryption key not initialized. Verify password first.")
        return self._key

    def encrypt(self, payload: bytes) -> bytes:
        fernet = Fernet(self._ensure_key())
        return fernet.encrypt(payload)

    def decrypt(self, token: bytes) -> bytes:
        fernet = Fernet(self._ensure_key())
        try:
            return fernet.decrypt(token)
        except InvalidToken as exc:
            raise ValueError("Invalid encryption token or incorrect password.") from exc


class ChatHistoryStore:
    """
    Stores encrypted conversations per model (role/content pairs).
    """

    def __init__(self, base_dir: str = "history", security: Optional["SecurityManager"] = None):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        if security is None:
            raise ValueError("ChatHistoryStore requires a SecurityManager instance.")
        self.security = security

    def _path_for_model(self, model_name: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
        return os.path.join(self.base_dir, f"{sanitized}.chat")

    def load(self, model_name: str) -> List[Dict[str, Any]]:
        path = self._path_for_model(model_name)
        if not os.path.exists(path):
            return []
        with open(path, "rb") as fh:
            token = fh.read()
        raw = self.security.decrypt(token)
        try:
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def save(self, model_name: str, messages: List[Dict[str, Any]]) -> None:
        payload = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        token = self.security.encrypt(payload)
        path = self._path_for_model(model_name)
        with open(path, "wb") as fh:
            fh.write(token)

    def clear(self, model_name: str) -> None:
        path = self._path_for_model(model_name)
        if os.path.exists(path):
            os.remove(path)
