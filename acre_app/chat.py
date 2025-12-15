import re
import tkinter as tk
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk

from . import global_state as gs
from .constants import OUTPUTS_PATH
from .ui_helpers import update_logo_visibility, update_status

THINK_TAG_PATTERN = re.compile(r"\s*/(no_)?think\s*$", re.IGNORECASE)
IMAGE_MARKER_PATTERN = re.compile(r"\[\[image:(.+?)\]\]")
DOC_MARKER_PATTERN = re.compile(r"\[\[doc:(.+?)\]\]")


def _strip_tags(text: str) -> str:
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()
    return cleaned or text


def render_history() -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.delete("1.0", tk.END)
    if gs.mgr and gs.mgr.is_loaded() and not gs.mgr.is_image_backend():
        for message in gs.mgr.get_history():
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                content = _strip_tags(str(content))
            else:
                content = str(content)

            images, docs, cleaned = _extract_attachments(content)

            if role == "system":
                prefix = "System"
            elif role == "assistant":
                prefix = "Assistant"
            else:
                prefix = "User"

            gs.chat_history.insert(tk.END, f"{prefix}:")
            if images or docs:
                gs.chat_history.insert(tk.END, "\n")
            else:
                gs.chat_history.insert(tk.END, " ")

            if cleaned.strip():
                gs.chat_history.insert(tk.END, cleaned.strip())
                gs.chat_history.insert(tk.END, "\n")

            for rel in docs:
                path = _resolve_user_output_path(rel)
                name = path.name if path else rel
                gs.chat_history.insert(tk.END, f"[Document] {name}\n")

            for rel in images:
                path = _resolve_user_output_path(rel)
                if path and path.exists():
                    _insert_image_inline(str(path))
                    gs.chat_history.insert(tk.END, "\n")
                else:
                    gs.chat_history.insert(tk.END, f"[Image missing] {rel}\n")

            gs.chat_history.insert(tk.END, "\n")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()


def append_user_message(text: str) -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, f"User: {text}\n\n")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()


def append_assistant_message(text: str) -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, f"Assistant: {text}\n\n")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()


def start_assistant_stream() -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, "Assistant: ")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()


def append_assistant_stream_chunk(text: str) -> None:
    if not text or gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, text)
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()


def end_assistant_stream() -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, "\n\n")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()

def append_user_attachment_message(text: str, *, image_path: str | None = None, doc_path: str | None = None) -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, "User:\n")
    if image_path:
        _insert_image_inline(image_path)
        gs.chat_history.insert(tk.END, "\n")
    if doc_path:
        gs.chat_history.insert(tk.END, f"[Document] {Path(doc_path).name}\n")
    if text.strip():
        gs.chat_history.insert(tk.END, f"{text.strip()}\n")
    gs.chat_history.insert(tk.END, "\n")
    gs.chat_history.see(tk.END)
    gs.chat_history.configure(state="disabled")
    update_logo_visibility()

def _resolve_user_output_path(relative: str) -> Optional[Path]:
    rel = (relative or "").strip()
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate
    user_bucket = gs.current_user or ""
    base = OUTPUTS_PATH / user_bucket if user_bucket else OUTPUTS_PATH
    return (base / rel).resolve()


def _extract_attachments(content: str) -> tuple[list[str], list[str], str]:
    images: list[str] = []
    docs: list[str] = []
    if not content:
        return images, docs, ""

    def _collect(pattern: re.Pattern, target: list[str], text: str) -> str:
        for match in pattern.finditer(text):
            value = (match.group(1) or "").strip()
            if value:
                target.append(value)
        return pattern.sub("", text)

    cleaned = content
    cleaned = _collect(IMAGE_MARKER_PATTERN, images, cleaned)
    cleaned = _collect(DOC_MARKER_PATTERN, docs, cleaned)
    return images, docs, cleaned


def _insert_image_inline(path: str) -> None:
    if gs.chat_history is None:
        return
    try:
        with Image.open(path) as image:
            width = 560
            width = max(1, width)
            height = int(image.height * (width / max(1, image.width)))
            preview = image.resize((width, height))
            tk_image = ImageTk.PhotoImage(preview)
        gs.chat_images.append(tk_image)
        if len(gs.chat_images) > 16:
            gs.chat_images.pop(0)
        gs.chat_history.image_create(tk.END, image=tk_image)
    except Exception as exc:
        update_status(f"Preview failed: {exc}")


def insert_image_preview(path: str, *, prefix: str = "Assistant") -> None:
    if gs.chat_history is None:
        return
    gs.chat_history.configure(state="normal")
    gs.chat_history.insert(tk.END, f"{prefix}:\n")
    _insert_image_inline(path)
    gs.chat_history.insert(tk.END, f"\n\n")
    gs.chat_history.configure(state="disabled")
    gs.chat_history.see(tk.END)
