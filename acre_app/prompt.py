import re
import threading
import tkinter as tk

from . import global_state as gs
from .chat import append_user_message, render_history
from .constants import OUTPUTS_PATH, PLACEHOLDER
from .gallery import refresh_gallery
from .settings import get_prefs
from .ui_helpers import update_status

THINK_TAG_PATTERN = re.compile(r"\s*/(no_)?think\s*$", re.IGNORECASE)


def _display_prompt_text(text: str) -> str:
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()
    return cleaned or text


def run_prompt() -> None:
    if gs.entry is None:
        return
    text = gs.entry.get("1.0", tk.END).strip()
    if not text or text == PLACEHOLDER:
        return
    if not gs.mgr or not gs.mgr.is_loaded():
        update_status("Load a model before sending a prompt.")
        return
    # Check if generation is already in progress
    if hasattr(gs.mgr, "_generating") and gs.mgr._generating:
        update_status("Generation in progress. Please wait...")
        return
    gs.entry.delete("1.0", tk.END)
    if hasattr(gs.mgr, "is_tts_backend") and gs.mgr.is_tts_backend():
        update_status("Synthesizing audio...")
        output_dir = OUTPUTS_PATH / (gs.current_user or "")

        def worker_tts(prompt_text: str) -> None:
            error = None
            path = None
            try:
                path = gs.mgr.run_tts(prompt_text, outdir=output_dir)
            except Exception as exc:
                error = str(exc)

            def done() -> None:
                if error:
                    update_status(error)
                    return
                if gs.chat_history is None:
                    return
                gs.chat_history.configure(state="normal")
                gs.chat_history.insert("end", f"Assistant (TTS): saved to {path}\n\n")
                gs.chat_history.configure(state="disabled")
                gs.chat_history.see("end")
                update_status("Done")

            gs.root.after(0, done)

        threading.Thread(target=worker_tts, args=(text,), daemon=True).start()
        return
    if not gs.mgr.is_image_backend():
        append_user_message(_display_prompt_text(text))
        update_status("Generating...")

        def worker_text(prompt_text: str) -> None:
            error = None
            try:
                gs.mgr.generate(prompt_text)
            except RuntimeError as exc:
                # Handle concurrent generation attempts gracefully
                if "already in progress" in str(exc):
                    error = "Please wait for the current generation to complete before sending another query."
                else:
                    error = str(exc)
            except Exception as exc:
                error = str(exc)

            def done_text() -> None:
                update_status(error if error else "Done")
                render_history()

            gs.root.after(0, done_text)

        threading.Thread(target=worker_text, args=(text,), daemon=True).start()
        return
    update_status("Generating image...")
    output_dir = OUTPUTS_PATH / (gs.current_user or "")
    prefs = get_prefs()

    def worker_image(prompt_text: str) -> None:
        error = None
        path = None
        try:
            path = gs.mgr.generate_image(
                prompt_text,
                steps=prefs["image_steps"],
                guidance=prefs["image_guidance"],
                width=prefs["image_width"],
                height=prefs["image_height"],
                seed=prefs["image_seed"],
                outdir=output_dir,
            )
        except Exception as exc:
            error = str(exc)

        def done_image() -> None:
            if error:
                update_status(error)
                return
            update_status(f"Saved: {path}")
            if gs.gallery_container:
                try:
                    refresh_gallery(gs.gallery_container)
                except Exception:
                    pass

        gs.root.after(0, done_image)

    threading.Thread(target=worker_image, args=(text,), daemon=True).start()
