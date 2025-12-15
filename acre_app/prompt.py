import re
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

from . import global_state as gs
from . import paths
from .chat import (
    append_user_message,
    append_assistant_message,
    append_assistant_stream_chunk,
    append_user_attachment_message,
    end_assistant_stream,
    render_history,
    start_assistant_stream,
)
from .constants import OUTPUTS_PATH, PLACEHOLDER
from .gallery import refresh_gallery
from .settings import get_prefs
from .ui_helpers import update_status
from .safety import find_trigger_terms, build_safety_message


THINK_TAG_PATTERN = re.compile(r"\s*/(no_)?think\s*$", re.IGNORECASE)


def _display_prompt_text(text: str) -> str:
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()
    return cleaned or text


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _log_benchmark(prompt_text: str, response_text: str, duration: float) -> None:
    user_bucket = gs.current_user or "_global_"
    bench_dir = OUTPUTS_PATH / user_bucket
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench_file = bench_dir / "benchmarks.txt"
    model = getattr(gs.mgr, "_current_model_name", "unknown")
    backend = getattr(gs.mgr, "backend", "unknown")
    device = getattr(gs.mgr, "_device_pref", "unknown")
    tokens = _estimate_tokens(response_text)
    tps = tokens / duration if duration > 0 else 0.0
    entry = (
        f"Model: {model}\n"
        f"Backend: {backend}\n"
        f"Device: {device}\n"
        f"Duration_sec: {duration:.3f}\n"
        f"Tokens_est: {tokens}\n"
        f"TPS_est: {tps:.2f}\n"
        f"Prompt_chars: {len(prompt_text)}\n"
        f"Response_chars: {len(response_text)}\n"
        "----\n"
    )
    bench_file.write_text(
        bench_file.read_text(encoding="utf-8") + entry if bench_file.exists() else entry,
        encoding="utf-8",
    )


def _set_generation_controls(running: bool, *, allow_cancel: bool = True) -> None:
    try:
        if gs.send_button is not None:
            gs.send_button.configure(state="disabled" if running else "normal")
    except Exception:
        pass
    try:
        if gs.stop_button is None:
            return
        if running and allow_cancel:
            if not gs.stop_button.winfo_manager():
                gs.stop_button.pack(side="right", padx=(0, 6), pady=4)
            gs.stop_button.configure(state="normal")
        else:
            gs.stop_button.configure(state="disabled")
            try:
                gs.stop_button.pack_forget()
            except Exception:
                pass
    except Exception:
        pass


def run_prompt() -> None:
    if gs.entry is None:
        return
    text = gs.entry.get("1.0", tk.END).strip()
    if not text or text == PLACEHOLDER:
        return

    triggered = find_trigger_terms(text)
    if triggered:
        gs.entry.delete("1.0", tk.END)

        safety_msg = build_safety_message(triggered)

        append_user_message(text)

        if gs.chat_history:

            gs.chat_history.configure(state="normal")
            gs.chat_history.insert("end", f"Assistant (Safety Notice): {safety_msg}\n\n")
            gs.chat_history.configure(state="disabled")
            gs.chat_history.see("end")

        update_status("Blocked due to safety filters.")
        return

    if not gs.mgr or not gs.mgr.is_loaded():
        update_status("Load a model before sending a prompt.")
        return
    if hasattr(gs.mgr, "_generating") and gs.mgr._generating:
        update_status("Generation in progress. Please wait...")
        return

    pending_image = getattr(gs, "pending_image_path", None)
    if pending_image and hasattr(gs.mgr, "is_vision_backend") and gs.mgr.is_vision_backend():
        question = _display_prompt_text(text)
        gs.entry.delete("1.0", tk.END)
        update_status("Analyzing image…")
        _set_generation_controls(True, allow_cancel=False)
        gs.active_cancel_event = None

        try:
            base = OUTPUTS_PATH / (gs.current_user or "")
            rel = str(Path(str(pending_image)).resolve().relative_to(base.resolve()))
        except Exception:
            rel = str(Path(str(pending_image)).name)
        marker = f"[[image:{rel}]]"
        user_content = f"{marker}\n{question}"

        if getattr(gs.mgr, "_history_enabled", True):
            try:
                gs.mgr.add_history_entry("user", user_content)
            except Exception:
                pass
            render_history()
        else:
            append_user_attachment_message(question, image_path=str(pending_image))

        def worker() -> None:
            error: str | None = None
            answer: str | None = None
            try:
                answer = gs.mgr.analyze_image(str(pending_image), question)
            except Exception as exc:
                error = str(exc)

            def done() -> None:
                if error:
                    update_status(f"Image analysis failed: {error}")
                    _set_generation_controls(False)
                    return

                if getattr(gs.mgr, "_history_enabled", True):
                    try:
                        gs.mgr.add_history_entry("assistant", str(answer or ""))
                    except Exception:
                        pass
                    render_history()
                else:
                    append_assistant_message(str(answer or ""))
                update_status("Done")
                _set_generation_controls(False)

            if gs.root:
                gs.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()
        return

    pending_doc = getattr(gs, "pending_doc_path", None)
    if pending_doc and not gs.mgr.is_image_backend():
        if (hasattr(gs.mgr, "is_ocr_backend") and gs.mgr.is_ocr_backend()) or (
            hasattr(gs.mgr, "is_asr_backend") and gs.mgr.is_asr_backend()
        ) or (hasattr(gs.mgr, "is_tts_backend") and gs.mgr.is_tts_backend()):
            pending_doc = None

    if pending_doc:
        question = _display_prompt_text(text)
        gs.entry.delete("1.0", tk.END)
        update_status("Reading document…")
        _set_generation_controls(True, allow_cancel=True)

        cancel_event = threading.Event()
        gs.active_cancel_event = cancel_event
        chunk_queue: queue.Queue[str] = queue.Queue()
        result: dict[str, object] = {
            "error": None,
            "response_text": "",
            "duration": 0.0,
            "canceled": False,
        }
        done_event = threading.Event()

        try:
            base = OUTPUTS_PATH / (gs.current_user or "")
            rel = str(Path(str(pending_doc)).resolve().relative_to(base.resolve()))
        except Exception:
            rel = str(Path(str(pending_doc)).name)
        marker = f"[[doc:{rel}]]"
        user_content = f"{marker}\n{question}"

        if getattr(gs.mgr, "_history_enabled", True):
            try:
                gs.mgr.add_history_entry("user", user_content)
            except Exception:
                pass
            render_history()
        else:
            append_user_attachment_message(question, doc_path=str(pending_doc))

        start_assistant_stream()

        def worker_doc_qa(doc_path: str, question_text: str) -> None:
            error: str | None = None
            pieces: list[str] = []
            start_time = time.time()
            try:
                from .doc_qa import (
                    build_cited_prompt,
                    load_document_chunks,
                    select_relevant_excerpts,
                )
                prefs = get_prefs()
                chunks = load_document_chunks(
                    Path(doc_path),
                    cancel_event=cancel_event,
                    models_dir=paths.models_dir(),
                    device_pref=str(prefs.get("device_preference", "auto")),
                )
                excerpts = select_relevant_excerpts(chunks, question_text, max_excerpts=6)
                prompt_text = build_cited_prompt(Path(doc_path).name, question_text, excerpts)
                if gs.root:
                    gs.root.after(0, lambda: update_status("Generating…"))

                messages: list[dict] = []
                system_prompt = str(prefs.get("system_prompt", "") or "").strip()
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt_text})
                for chunk in gs.mgr.generate_stream_messages(messages, cancel_event=cancel_event):
                    if not chunk:
                        continue
                    pieces.append(chunk)
                    chunk_queue.put(chunk)
            except RuntimeError as exc:
                message = str(exc)
                if "already in progress" in message:
                    error = "Please wait for the current generation to complete before sending another query."
                else:
                    error = message
            except Exception as exc:
                error = str(exc)
            duration = time.time() - start_time
            result["error"] = error
            result["response_text"] = "".join(pieces)
            result["duration"] = duration
            result["canceled"] = bool(cancel_event.is_set())
            done_event.set()

        def pump_queue_doc() -> None:
            pending_chunks: list[str] = []
            while True:
                try:
                    pending_chunks.append(chunk_queue.get_nowait())
                except queue.Empty:
                    break
            if pending_chunks:
                append_assistant_stream_chunk("".join(pending_chunks))

            if not done_event.is_set():
                if gs.root:
                    gs.root.after(50, pump_queue_doc)
                return

            # Final drain.
            pending_chunks = []
            while True:
                try:
                    pending_chunks.append(chunk_queue.get_nowait())
                except queue.Empty:
                    break
            if pending_chunks:
                append_assistant_stream_chunk("".join(pending_chunks))

            error = result.get("error")
            response_text = str(result.get("response_text") or "")
            canceled = bool(result.get("canceled"))

            error_text = str(error or "").strip()
            if canceled and (not error_text or "cancel" in error_text.lower()):
                if not response_text.strip():
                    append_assistant_stream_chunk("Stopped.")
                update_status("Stopped.")
                end_assistant_stream()
            elif error_text:
                if not response_text.strip():
                    append_assistant_stream_chunk(f"[Error] {error_text}")
                update_status(error_text)
                end_assistant_stream()
            else:
                update_status("Done")
                if getattr(gs.mgr, "_history_enabled", True):
                    try:
                        gs.mgr.add_history_entry("assistant", response_text)
                    except Exception:
                        pass
                    render_history()
                else:
                    end_assistant_stream()
            _set_generation_controls(False)
            gs.active_cancel_event = None

        threading.Thread(target=worker_doc_qa, args=(str(pending_doc), question), daemon=True).start()
        if gs.root:
            gs.root.after(0, pump_queue_doc)
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
        display_text = _display_prompt_text(text)
        append_user_message(display_text)
        start_assistant_stream()
        update_status("Generating...")
        _set_generation_controls(True, allow_cancel=True)

        cancel_event = threading.Event()
        gs.active_cancel_event = cancel_event
        chunk_queue: queue.Queue[str] = queue.Queue()
        result: dict[str, object] = {
            "error": None,
            "response_text": "",
            "duration": 0.0,
            "canceled": False,
        }
        done_event = threading.Event()

        def worker_text_stream(prompt_text: str) -> None:
            error: str | None = None
            pieces: list[str] = []
            start_time = time.time()
            try:
                for chunk in gs.mgr.generate_stream(prompt_text, cancel_event=cancel_event):
                    if not chunk:
                        continue
                    pieces.append(chunk)
                    chunk_queue.put(chunk)
            except RuntimeError as exc:
                message = str(exc)
                if "already in progress" in message:
                    error = "Please wait for the current generation to complete before sending another query."
                else:
                    error = message
            except Exception as exc:
                error = str(exc)
            duration = time.time() - start_time
            result["error"] = error
            result["response_text"] = "".join(pieces)
            result["duration"] = duration
            result["canceled"] = bool(cancel_event.is_set())
            done_event.set()

        def pump_queue() -> None:
            pending: list[str] = []
            while True:
                try:
                    pending.append(chunk_queue.get_nowait())
                except queue.Empty:
                    break
            if pending:
                append_assistant_stream_chunk("".join(pending))

            if not done_event.is_set():
                if gs.root:
                    gs.root.after(50, pump_queue)
                return

            # Final drain (in case chunks landed after the last tick).
            pending = []
            while True:
                try:
                    pending.append(chunk_queue.get_nowait())
                except queue.Empty:
                    break
            if pending:
                append_assistant_stream_chunk("".join(pending))

            error = result.get("error")
            response_text = str(result.get("response_text") or "")
            canceled = bool(result.get("canceled"))
            duration = float(result.get("duration") or 0.0)

            if error:
                if not response_text.strip():
                    append_assistant_stream_chunk(f"[Error] {error}")
                update_status(str(error))
                end_assistant_stream()
            elif canceled:
                if not response_text.strip():
                    append_assistant_stream_chunk("Stopped.")
                update_status("Stopped.")
                end_assistant_stream()
            else:
                update_status("Done")
                if getattr(gs.mgr, "_history_enabled", True):
                    render_history()
                else:
                    end_assistant_stream()
                try:
                    _log_benchmark(prompt_text, response_text, duration)
                except Exception:
                    pass
            _set_generation_controls(False)
            gs.active_cancel_event = None

        threading.Thread(target=worker_text_stream, args=(text,), daemon=True).start()
        if gs.root:
            gs.root.after(0, pump_queue)
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
