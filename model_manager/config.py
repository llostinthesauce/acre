from dataclasses import dataclass
from typing import Tuple


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    stop: Tuple[str, ...] = ("User:",)
