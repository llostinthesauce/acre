import json
import re
from pathlib import Path
from typing import List


GUARDRAIL_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "guardrail_list.json"
)

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_guardrails() -> List[str]:
    try:
        with open(GUARDRAIL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    flat_terms = []
    if isinstance(data, dict):
        for items in data.values():
            if isinstance(items, list):
                for term in items:
                    t = _normalize(str(term))
                    if t:
                        flat_terms.append(t)
    elif isinstance(data, list):
        flat_terms = [_normalize(str(x)) for x in data]

    return flat_terms


GUARDRAIL_TERMS = load_guardrails()


def find_trigger_terms(text: str) -> List[str]:
    normalized = _normalize(text)
    return [term for term in GUARDRAIL_TERMS if term in normalized]


def build_safety_message(triggered: List[str]) -> str:
    joined = ", ".join(f"“{t}”" for t in triggered)
    return (
        "Your request has been blocked because it contains restricted or dangerous "
        f"content (triggered terms: {joined})."
    )
