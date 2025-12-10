import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from . import global_state as gs
from .chat import append_assistant_message, append_user_message, render_history
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BUTTON_RADIUS,
    FONT_BOLD,
    FONT_UI,
    MUTED,
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
            text="Vision: analyze documents (image analysis disabled)",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
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
            text="Document: choose PDF/text → summary",
            font=FONT_UI,
            text_color=TEXT,
        ).pack(side="left", padx=12, pady=10)
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
    update_status("Image analysis is disabled for this demo.")
