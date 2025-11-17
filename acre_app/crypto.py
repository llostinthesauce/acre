from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import pbkdf2_hmac

from cryptography.fernet import Fernet, InvalidToken


def _decode_b64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("utf-8"))


def derive_fernet_key(password: str, *, salt_b64: str, iterations: int = 200_000) -> bytes:
    if not password:
        raise ValueError("Password required for key derivation.")
    salt = _decode_b64(salt_b64)
    key = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations), dklen=32)
    return base64.urlsafe_b64encode(key)


@dataclass
class ChatEncryptor:

    key: bytes

    def __post_init__(self) -> None:
        self._fernet = Fernet(self.key)

    def encrypt(self, payload: bytes) -> bytes:
        return self._fernet.encrypt(payload)

    def decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)


__all__ = ["ChatEncryptor", "derive_fernet_key", "InvalidToken"]
