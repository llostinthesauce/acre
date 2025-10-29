import re
import tkinter as tk

from PIL import Image, ImageTk

from . import global_state as gs
from .ui_helpers import update_logo_visibility, update_status

THINK_TAG_PATTERN = re.compile(r"\s*/(no_)?think\s*$", re.IGNORECASE)


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
                content = _strip_tags(content)
            prefix = "User" if role == "user" else "Assistant"
            gs.chat_history.insert(tk.END, f"{prefix}: {content}\n\n")
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


def insert_image_preview(path: str) -> None:
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
        gs.chat_history.configure(state="normal")
        gs.chat_history.insert(tk.END, "Assistant: ")
        gs.chat_history.image_create(tk.END, image=tk_image)
        gs.chat_history.insert(tk.END, f"\nSaved to: {path}\n\n")
        gs.chat_history.configure(state="disabled")
        gs.chat_history.see(tk.END)
    except Exception as exc:
        update_status(f"Preview failed: {exc}")
