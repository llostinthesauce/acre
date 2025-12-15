import threading
import time
import secrets
import shutil
from pathlib import Path
from tkinter import filedialog, simpledialog

import customtkinter as ctk
from PIL import Image, ImageGrab

from . import global_state as gs
from .chat import append_assistant_message, append_user_message, render_history
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BUTTON_RADIUS,
    FONT_BOLD,
    FONT_UI,
    MUTED,
    OUTPUTS_PATH,
    PLACEHOLDER,
    SURFACE_ELEVATED,
    TEXT,
)
from .documents import summarize_document
from .ui_helpers import update_status


def _build_button(parent, text, command) -> None:
    ctk.CTkButton(
        parent,
        text=text,
        command=command,
        font=FONT_BOLD,
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
    ).pack(side="left", padx=8, pady=8)


def refresh_attach_row() -> None:
    if gs.attach_row is None:
        return
    for widget in gs.attach_row.winfo_children():
        try:
            widget.destroy()
        except Exception:
            pass
    if not gs.mgr or not gs.mgr.is_loaded():
        ctk.CTkLabel(
            gs.attach_row,
            text="Load a model to enable attachments.",
            font=FONT_UI,
            text_color=MUTED,
        ).pack(side="left", padx=12, pady=10)
        return
    if hasattr(gs.mgr, "is_ocr_backend") and gs.mgr.is_ocr_backend():
        ctk.CTkLabel(
            gs.attach_row, text="OCR: choose image → text", font=FONT_UI, text_color=TEXT
        ).pack(side="left", padx=12, pady=10)
        _build_button(gs.attach_row, "Choose Image…", do_ocr)
    elif hasattr(gs.mgr, "is_asr_backend") and gs.mgr.is_asr_backend():
        ctk.CTkLabel(
            gs.attach_row, text="ASR: choose audio → text", font=FONT_UI, text_color=TEXT
        ).pack(side="left", padx=12, pady=10)
        _build_button(gs.attach_row, "Choose Audio…", do_asr)
    elif hasattr(gs.mgr, "is_tts_backend") and gs.mgr.is_tts_backend():
        ctk.CTkLabel(
            gs.attach_row,
            text="TTS: type text → audio (.wav)",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
    elif hasattr(gs.mgr, "is_vision_backend") and gs.mgr.is_vision_backend():
        ctk.CTkLabel(
            gs.attach_row,
            text="Vision: attach an image, then ask in chat",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
        if getattr(gs, "pending_image_path", None):
            name = Path(str(gs.pending_image_path)).name
            ctk.CTkLabel(
                gs.attach_row,
                text=f"Attached: {name}",
                font=FONT_UI,
                text_color=MUTED,
            ).pack(side="left", padx=8, pady=10)
            _build_button(gs.attach_row, "Clear", clear_pending_image)
        _build_button(gs.attach_row, "Attach Image…", attach_image)
        _build_button(gs.attach_row, "Paste Image", paste_image)
        _build_button(gs.attach_row, "Analyze Document…", analyze_document)
    elif hasattr(gs.mgr, "is_image_backend") and gs.mgr.is_image_backend():
        ctk.CTkLabel(
            gs.attach_row,
            text="Diffusion: type prompt → image",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
    else:
        ctk.CTkLabel(
            gs.attach_row,
            text="Docs: attach a document, then ask in chat (citations)",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
        if getattr(gs, "pending_doc_path", None):
            name = Path(str(gs.pending_doc_path)).name
            ctk.CTkLabel(
                gs.attach_row,
                text=f"Attached: {name}",
                font=FONT_UI,
                text_color=MUTED,
            ).pack(side="left", padx=8, pady=10)
            _build_button(gs.attach_row, "Clear", clear_pending_doc)
        _build_button(gs.attach_row, "Attach Document…", attach_document)
        _build_button(gs.attach_row, "Analyze Document…", analyze_document)


def do_ocr() -> None:
    if not gs.mgr or not hasattr(gs.mgr, "is_ocr_backend") or not gs.mgr.is_ocr_backend():
        update_status("Load an OCR model first.")
        return
    path = filedialog.askopenfilename(
        title="Choose image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")],
    )
    if not path:
        return
    update_status("Running OCR…")

    def worker() -> None:
        try:
            text = gs.mgr.run_ocr(path)
            append_assistant_message(f"OCR Result: {text}")
            if getattr(gs.mgr, "add_history_entry", None):
                gs.mgr.add_history_entry("assistant", f"OCR Result: {text}")
            render_history()
            update_status("Done")
        except Exception as exc:
            update_status(str(exc))

    threading.Thread(target=worker, daemon=True).start()


def do_asr() -> None:
    if not gs.mgr or not hasattr(gs.mgr, "is_asr_backend") or not gs.mgr.is_asr_backend():
        update_status("Load an ASR model first.")
        return
    path = filedialog.askopenfilename(
        title="Choose audio",
        filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.opus")],
    )
    if not path:
        return
    update_status("Transcribing audio…")

    def worker() -> None:
        try:
            text = gs.mgr.run_asr(path)
            append_assistant_message(f"Transcription: {text}")
            if getattr(gs.mgr, "add_history_entry", None):
                gs.mgr.add_history_entry("assistant", f"Transcription: {text}")
            render_history()
            update_status("Done")
        except Exception as exc:
            update_status(str(exc))

    threading.Thread(target=worker, daemon=True).start()


def analyze_document() -> None:
    if not gs.mgr or not gs.mgr.is_loaded():
        update_status("Load a compatible model first.")
        return
    if hasattr(gs.mgr, "is_image_backend") and gs.mgr.is_image_backend():
        update_status("Switch to a text-capable model to analyze documents.")
        return
    # Block obvious non-generative pipelines.
    if (hasattr(gs.mgr, "is_ocr_backend") and gs.mgr.is_ocr_backend()) or (
        hasattr(gs.mgr, "is_asr_backend") and gs.mgr.is_asr_backend()
    ) or (hasattr(gs.mgr, "is_tts_backend") and gs.mgr.is_tts_backend()):
        update_status("Load a text-capable model to analyze documents.")
        return
    path = filedialog.askopenfilename(
        title="Choose document",
        filetypes=[
            ("Supported", "*.pdf *.txt *.md *.markdown"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt *.md *.markdown"),
        ],
    )
    if not path:
        return
    doc_path = Path(path)
    message = f"Analyzing document: {doc_path.name}"
    append_user_message(message)
    if getattr(gs.mgr, "add_history_entry", None):
        gs.mgr.add_history_entry("user", message)
    render_history()
    update_status("Reading document…")

    def worker() -> None:
        try:
            summary = summarize_document(gs.mgr, doc_path)
        except Exception as exc:
            update_status(f"Document analysis failed: {exc}")
            return

        def done() -> None:
            append_assistant_message(summary)
            if getattr(gs.mgr, "add_history_entry", None):
                gs.mgr.add_history_entry("assistant", summary)
            render_history()
            update_status("Document analyzed.")

        if gs.root:
            gs.root.after(0, done)

    threading.Thread(target=worker, daemon=True).start()


def analyze_image() -> None:
    if not gs.mgr or not hasattr(gs.mgr, "is_vision_backend") or not gs.mgr.is_vision_backend():
        update_status("Load a vision model first.")
        return
    path = filedialog.askopenfilename(
        title="Choose image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")],
    )
    if not path:
        return
    question = simpledialog.askstring(
        "Image question",
        "What would you like to ask about this image?",
        initialvalue="Describe this image.",
        parent=gs.root,
    )
    question = (question or "").strip()
    if not question:
        return

    image_path = Path(path)
    message = f"Image Q/A: {image_path.name}\nQuestion: {question}"
    append_user_message(message)
    if getattr(gs.mgr, "add_history_entry", None):
        gs.mgr.add_history_entry("user", message)
    render_history()
    update_status("Analyzing image…")

    def worker() -> None:
        try:
            answer = gs.mgr.analyze_image(str(image_path), question)
        except Exception as exc:
            update_status(f"Image analysis failed: {exc}")
            return

        def done() -> None:
            append_assistant_message(answer)
            if getattr(gs.mgr, "add_history_entry", None):
                gs.mgr.add_history_entry("assistant", answer)
            render_history()
            update_status("Image analyzed.")

        if gs.root:
            gs.root.after(0, done)

    threading.Thread(target=worker, daemon=True).start()


def _user_attachments_dir() -> Path:
    bucket = gs.current_user or ""
    root = OUTPUTS_PATH / bucket if bucket else OUTPUTS_PATH
    out = root / "attachments"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _store_image_file(src: Path) -> Path:
    ext = src.suffix.lower() if src.suffix else ".png"
    safe_ext = ext if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} else ".png"
    name = f"img_{int(time.time())}_{secrets.token_hex(4)}{safe_ext}"
    dest = _user_attachments_dir() / name
    shutil.copy2(src, dest)
    return dest


def _store_clipboard_image(image: Image.Image) -> Path:
    name = f"clip_{int(time.time())}_{secrets.token_hex(4)}.png"
    dest = _user_attachments_dir() / name
    image.save(dest, format="PNG")
    return dest


def attach_image() -> None:
    if not gs.mgr or not hasattr(gs.mgr, "is_vision_backend") or not gs.mgr.is_vision_backend():
        update_status("Load a vision model first.")
        return
    path = filedialog.askopenfilename(
        title="Choose image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")],
    )
    if not path:
        return
    try:
        dest = _store_image_file(Path(path))
    except Exception as exc:
        update_status(f"Failed to attach image: {exc}")
        return
    gs.pending_image_path = str(dest)
    update_status(f"Attached image: {dest.name}")
    refresh_attach_row()


def paste_image() -> None:
    if not gs.mgr or not hasattr(gs.mgr, "is_vision_backend") or not gs.mgr.is_vision_backend():
        update_status("Load a vision model first.")
        return
    if not try_attach_clipboard_image(quiet=False):
        update_status("Clipboard does not contain an image.")


def try_attach_clipboard_image(*, quiet: bool = True) -> bool:
    if not gs.mgr or not hasattr(gs.mgr, "is_vision_backend") or not gs.mgr.is_vision_backend():
        return False
    try:
        grabbed = ImageGrab.grabclipboard()
    except Exception as exc:
        if not quiet:
            update_status(f"Clipboard unavailable: {exc}")
        return False
    if isinstance(grabbed, list) and grabbed:
        candidate = Path(str(grabbed[0]))
        if candidate.exists() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            try:
                dest = _store_image_file(candidate)
            except Exception as exc:
                if not quiet:
                    update_status(f"Failed to attach image: {exc}")
                return False
            gs.pending_image_path = str(dest)
            if not quiet:
                update_status(f"Attached image: {dest.name}")
            refresh_attach_row()
            return True
        return False
    if not isinstance(grabbed, Image.Image):
        return False
    try:
        dest = _store_clipboard_image(grabbed.convert("RGB"))
    except Exception as exc:
        if not quiet:
            update_status(f"Failed to save pasted image: {exc}")
        return False
    gs.pending_image_path = str(dest)
    if not quiet:
        update_status(f"Attached image: {dest.name}")
    refresh_attach_row()
    return True


def clear_pending_image() -> None:
    gs.pending_image_path = None
    update_status("Cleared attached image.")
    refresh_attach_row()


def _store_doc_file(src: Path) -> Path:
    ext = src.suffix.lower() if src.suffix else ".txt"
    safe_ext = ext if ext in {".pdf", ".txt", ".md", ".markdown"} else ".txt"
    name = f"doc_{int(time.time())}_{secrets.token_hex(4)}{safe_ext}"
    dest = _user_attachments_dir() / name
    shutil.copy2(src, dest)
    return dest


def attach_document() -> None:
    if not gs.mgr or not gs.mgr.is_loaded():
        update_status("Load a compatible model first.")
        return
    if hasattr(gs.mgr, "is_image_backend") and gs.mgr.is_image_backend():
        update_status("Switch to a text-capable model to analyze documents.")
        return
    if (hasattr(gs.mgr, "is_ocr_backend") and gs.mgr.is_ocr_backend()) or (
        hasattr(gs.mgr, "is_asr_backend") and gs.mgr.is_asr_backend()
    ) or (hasattr(gs.mgr, "is_tts_backend") and gs.mgr.is_tts_backend()):
        update_status("Load a text-capable model to analyze documents.")
        return
    path = filedialog.askopenfilename(
        title="Choose document",
        filetypes=[
            ("Supported", "*.pdf *.txt *.md *.markdown"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt *.md *.markdown"),
        ],
    )
    if not path:
        return
    try:
        dest = _store_doc_file(Path(path))
    except Exception as exc:
        update_status(f"Failed to attach document: {exc}")
        return
    gs.pending_doc_path = str(dest)
    update_status(f"Attached document: {dest.name}")
    refresh_attach_row()


def clear_pending_doc() -> None:
    gs.pending_doc_path = None
    update_status("Cleared attached document.")
    refresh_attach_row()
