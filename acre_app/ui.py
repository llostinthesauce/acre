import gc
import os
import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
import json
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from platform_utils import is_jetson
from . import global_state as gs
from .attachments import refresh_attach_row
from .chat import render_history
from .crypto import ChatEncryptor, derive_fernet_key
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BG_GRAD_TOP,
    BG_LIST,
    BORDER_ACCENT,
    BUTTON_RADIUS,
    CONTROL_BG,
    CONTROL_BORDER,
    CRITICAL,
    CRITICAL_HOVER,
    FONT_BOLD,
    FONT_H1,
    FONT_H2,
    FONT_UI,
    GLASS_BG,
    MUTED,
    OUTPUTS_PATH,
    PANEL_ELEVATED,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    SUCCESS,
    SUCCESS_HOVER,
    SURFACE_ELEVATED,
    SURFACE_HOVER,
    SURFACE_PRIMARY,
    TEXT,
    TITLE_BAR_ACCENT,
    TITLE_BAR_COLOR,
    TITLE_BAR_HEIGHT,
    THEMES,
    switch_theme,
    CONFIG_PATH,
)
from .gallery import ensure_user_dirs, refresh_gallery
from .models import add_model, pick_model, refresh_list, rename_model
from .training import open_training_dialog
from .prompt import run_prompt
from .settings import (
    clear_remember_me,
    ensure_users_bucket,
    ensure_encryption_metadata,
    get_active_user,
    get_prefs,
    get_remembered_user,
    list_usernames,
    load_settings,
    save_settings,
    set_active_user,
    set_credentials,
    set_prefs,
    set_disclaimer_ack,
    set_remember_me,
    verify_password,
)
from .ui_helpers import apply_native_font_scale, update_logo_visibility, update_status, recolor_whole_app
from .utils import open_path, to_float, to_int


def _refresh_theme_vars() -> None:
    # Rebind theme-dependent globals from constants so theme switches apply to rebuilt widgets.
    from . import constants as c
    g = globals()
    for name in [
        "ACCENT",
        "ACCENT_HOVER",
        "BG_GRAD_TOP",
        "BG_LIST",
        "BORDER_ACCENT",
        "CONTROL_BG",
        "CONTROL_BORDER",
        "CRITICAL",
        "CRITICAL_HOVER",
        "GLASS_BG",
        "MUTED",
        "PANEL_ELEVATED",
        "SURFACE_ELEVATED",
        "SURFACE_HOVER",
        "SURFACE_PRIMARY",
        "SUCCESS",
        "SUCCESS_HOVER",
        "TEXT",
        "TITLE_BAR_ACCENT",
        "TITLE_BAR_COLOR",
    ]:
        try:
            g[name] = getattr(c, name)
        except Exception:
            pass


def _clear_all_histories() -> None:
    base = Path("history") / (gs.current_user or "")
    if not base.exists():
        return
    if not messagebox.askyesno("Confirm", "Clear all histories for this user?"):
        return
    for path in base.glob("*.json"):
        try:
            path.unlink()
        except Exception:
            pass
    update_status("All histories cleared.")


def _clear_caches() -> None:
    mgr = gs.mgr
    if mgr:
        try:
            mgr.cleanup_memory()
            update_status("Caches cleared.")
            return
        except Exception:
            pass
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()
    update_status("Caches cleared.")


def _run_perf_test() -> None:
    if not gs.mgr:
        messagebox.showerror("Error", "Model manager not initialized.")
        return

    update_status("Running tiered performance test (see terminal for details)...")

    def worker():
        try:
            tiers = [
                {"n_ctx": 512, "max_tokens": 64},
                {"n_ctx": 1024, "max_tokens": 128},
                {"n_ctx": 2048, "max_tokens": 256},
            ]
            results = gs.mgr.run_perf_test_tiered(tiers=tiers)
            if not results:
                raise RuntimeError("No results returned from perf test.")

            def fmt(value: object, width: int) -> str:
                return str(value).rjust(width)

            def fmt_float(value: object, width: int, precision: int = 2) -> str:
                if isinstance(value, (int, float)):
                    return f"{value:.{precision}f}".rjust(width)
                return "n/a".rjust(width)

            header = "=== Performance Test (TinyLlama tiered) ==="
            first = results[0]
            print(header, flush=True)
            print(
                f"model: {first.get('model')} | threads: {first.get('n_threads')} | n_gpu_layers: {first.get('n_gpu_layers')}",
                flush=True,
            )
            columns = [
                ("pass", 5),
                ("ctx", 6),
                ("max", 5),
                ("load_s", 9),
                ("infer_s", 9),
                ("tokens", 8),
                ("eval_tps", 10),
                ("prompt_tps", 12),
                ("rssΔ_MB", 10),
                ("vram_MB", 12),
                ("power_W", 10),
                ("cpu_peak", 10),
            ]
            header_row = " ".join(col[0].rjust(col[1]) for col in columns)
            print(header_row, flush=True)
            print("-" * len(header_row), flush=True)

            vram_notes = set()
            for stats in results:
                eval_tps = stats.get("eval_tps")
                prompt_tps = stats.get("prompt_tps")
                rss_delta = stats.get("rss_delta_mb")
                vram_peak = stats.get("vram_mb", {}).get("max") if isinstance(stats.get("vram_mb"), dict) else None
                vram_reason = stats.get("vram_reason")
                if vram_reason:
                    vram_notes.add(vram_reason)
                power_avg = stats.get("power_w", {}).get("avg") if isinstance(stats.get("power_w"), dict) else None
                cpu_peak = stats.get("cpu_proc_pct", {}).get("max") if isinstance(stats.get("cpu_proc_pct"), dict) else None
                tier_idx = stats.get("tier") or 1
                tier_total = stats.get("tier_total") or len(results)
                row = " ".join(
                    [
                        fmt(f"{tier_idx}/{tier_total}", 5),
                        fmt(stats.get("n_ctx"), 6),
                        fmt(stats.get("max_tokens"), 5),
                        fmt_float(stats.get("load_s"), 9, 3),
                        fmt_float(stats.get("infer_s"), 9, 3),
                        fmt(stats.get("completion_tokens"), 8),
                        fmt_float(eval_tps, 10, 2),
                        fmt_float(prompt_tps, 12, 2),
                        fmt_float(rss_delta, 10, 2),
                        fmt_float(vram_peak, 12, 2) if not vram_reason else fmt("n/a*", 12),
                        fmt_float(power_avg, 10, 2),
                        fmt_float(cpu_peak, 10, 1),
                    ]
                )
                print(row, flush=True)

            if vram_notes:
                print("Notes:", ", ".join(sorted(vram_notes)), flush=True)

            update_status(f"Performance test complete ({len(results)} passes). Details in terminal.")
        except Exception as exc:
            update_status("Performance test failed.")
            messagebox.showerror("Performance test failed", str(exc))

    threading.Thread(target=worker, daemon=True).start()


def build_title_bar() -> None:
    if not gs.root:
        return
    bar = ctk.CTkFrame(
        gs.root,
        height=TITLE_BAR_HEIGHT,
        fg_color=TITLE_BAR_COLOR,
        corner_radius=0,
    )
    bar.pack(fill="x", side="top")
    bar.pack_propagate(False)
    gs.title_bar = bar

    def start_move(event) -> None:
        gs.drag_offset = (event.x, event.y)

    def stop_move(_event=None) -> None:
        gs.drag_offset = None

    def do_move(event) -> None:
        if not gs.drag_offset or not gs.root:
            return
        x = event.x_root - gs.drag_offset[0]
        y = event.y_root - gs.drag_offset[1]
        gs.root.geometry(f"+{x}+{y}")

    title_label = ctk.CTkLabel(
        bar,
        text="ACRE LLM Switchboard",
        font=FONT_H2,
        text_color=TEXT,
    )
    title_label.pack(side="left", padx=16)

    

    for widget in (bar, title_label):
        widget.bind("<ButtonPress-1>", start_move)
        widget.bind("<ButtonRelease-1>", stop_move)
        widget.bind("<B1-Motion>", do_move)

    def _get_saved_theme_name() -> str:
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return data.get("prefs", {}).get("theme", "Blue")
        except Exception:
            return "Blue"

    theme_var = ctk.StringVar(value=_get_saved_theme_name())

    def on_theme_change(choice: str):
        switch_theme(choice)
        if gs.root:
            recolor_whole_app(gs.root)
        update_status(f"Theme set to {choice}")

    theme_picker = ctk.CTkOptionMenu(
        bar,
        values=list(THEMES.keys()),
        variable=theme_var,
        command=on_theme_change,
        font=FONT_UI,
        dropdown_font=FONT_UI,
        fg_color=CONTROL_BG,
    button_color=CONTROL_BG,
    button_hover_color=CONTROL_BORDER,
        text_color=MUTED,
        corner_radius=BUTTON_RADIUS,
    )
    theme_picker.pack(side="right", padx=(0, 10), pady=8)

    def minimize_app() -> None:
        gs.drag_offset = None
        if gs.root:
            gs.root.iconify()

    def settings_app() -> None:
        open_settings()

    def close_app() -> None:
        on_close()

    btn_kwargs = dict(
        width=42,
        height=28,
        corner_radius=BUTTON_RADIUS,
        font=FONT_BOLD,
        text_color=TEXT,
    )

    close_btn = ctk.CTkButton(
        bar,
        text="✕",
        command=close_app,
        fg_color=CRITICAL,
        hover_color=CRITICAL_HOVER,
        **btn_kwargs,
    )
    close_btn.pack(side="right", padx=(0, 14), pady=8)

    minimize_btn = ctk.CTkButton(
        bar,
        text="—",
        command=minimize_app,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        **btn_kwargs,
    )
    minimize_btn.pack(side="right", padx=(0, 10), pady=8)

    settings_btn = ctk.CTkButton(
        bar,
        text="⚙",
        command=settings_app,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        **btn_kwargs,
    )
    settings_btn.pack(side="right", padx=(0, 10), pady=8)


def clear_chat() -> None:
    if not gs.mgr or not gs.mgr.is_loaded():
        update_status("Load a model to clear its history.")
        return
    if not gs.mgr.is_image_backend():
        gs.mgr.clear_history()
        render_history()
    update_status("Chat cleared.")


def logout_action() -> None:
    try:
        if gs.mgr:
            gs.mgr.unload()
    except Exception:
        pass
    if gs.current_user:
        settings = load_settings()
        clear_remember_me(settings, gs.current_user)
    gs.current_user = None
    gs.pending_user = None
    gs.encryption_key = None
    teardown_main_ui()
    build_gate_ui()


def teardown_main_ui() -> None:
    widgets = [
        gs.side_frame,
        gs.chat_frame,
        gs.tabs,
        gs.toggle_btn,
        gs.settings_btn,
    ]
    for widget in widgets:
        try:
            if widget is not None:
                widget.destroy()
        except Exception:
            pass
    gs.side_frame = None
    gs.chat_frame = None
    gs.chat_history = None
    gs.listbox = None
    gs.entry = None
    gs.status = None
    gs.toggle_btn = None
    gs.settings_btn = None
    gs.logo_label = None
    gs.tabs = None
    gs.settings_tab = None
    gs.attach_row = None
    gs.gallery_container = None
    gs.status_scroll = None
    gs.user_label = None


def _make_settings_card(parent: ctk.CTkFrame, title: str, blurb: str, *, title_font=FONT_H2, blurb_font=FONT_UI) -> ctk.CTkFrame:
    card = ctk.CTkFrame(parent, fg_color=GLASS_BG, corner_radius=RADIUS_MD)
    card.pack(fill="x", padx=4, pady=6)
    ctk.CTkLabel(
        card, text=title, font=title_font, text_color=TEXT, anchor="w"
    ).pack(anchor="w", padx=16, pady=(12, 4))
    ctk.CTkLabel(
        card,
        text=blurb,
        font=blurb_font,
        text_color=MUTED,
        wraplength=540,
        justify="left",
    ).pack(anchor="w", padx=16, pady=(0, 12))
    body = ctk.CTkFrame(card, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
    return body


def render_settings_tab(tab) -> None:
    if tab is None:
        return

    for widget in tab.winfo_children():
        widget.destroy()

    prefs = get_prefs()
    scale = prefs.get("text_scale", prefs["ui_scale"])

    def _scale_font(font_tuple):
        family, size, *rest = font_tuple
        new_size = max(8, int(size * scale))
        return (family, new_size, *rest)

    font_ui = _scale_font(FONT_UI)
    font_bold = _scale_font(FONT_BOLD)
    font_h1 = _scale_font(FONT_H1)
    font_h2 = _scale_font(FONT_H2)

    scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=12, pady=12)

    ctk.CTkLabel(
        scroll, text="Settings", font=font_h1, text_color=TEXT
    ).pack(anchor="w", padx=4, pady=(4, 2))

    ctk.CTkLabel(
        scroll,
        text=(
            "Tune how ACRE behaves. Each section explains what the controls do "
            "so you can make changes with confidence."
        ),
        font=font_ui,
        text_color=MUTED,
        wraplength=560,
        justify="left",
    ).pack(anchor="w", padx=4, pady=(0, 12))

    appearance_body = _make_settings_card(
        scroll,
        "Appearance",
        "Pick a color theme and text size. Changes apply immediately and are saved for next time.",
        title_font=font_h2,
        blurb_font=font_ui,
    )

    current_theme = prefs.get("theme", "Blue")
    theme_var = tk.StringVar(value=current_theme)

    def on_theme_change(choice: str):
        try:
            set_prefs({"theme": choice})
        except Exception:
            pass
        switch_theme(choice)
        _refresh_theme_vars()
        if gs.root:
            recolor_whole_app(gs.root)
        update_status(f"Theme set to {choice}")
        refresh_main_ui()

    row = ctk.CTkFrame(appearance_body, fg_color="transparent")
    row.pack(fill="x", pady=(0, 6))

    ctk.CTkLabel(row, text="Theme", font=FONT_UI, text_color=TEXT).grid(
        row=0, column=0, padx=(0, 8), sticky="w"
    )

    ctk.CTkOptionMenu(
        row,
        values=list(THEMES.keys()),
        variable=theme_var,
        font=font_ui,
        dropdown_font=font_ui,
        fg_color=CONTROL_BG,
    button_color=CONTROL_BG,
    button_hover_color=CONTROL_BORDER,
        text_color=MUTED,
        corner_radius=BUTTON_RADIUS,
        command=on_theme_change,
    ).grid(row=0, column=1, sticky="w")
    row.grid_columnconfigure(1, weight=1)

    text_scale_var = tk.DoubleVar(value=prefs["text_scale"])

    ui_text_row = ctk.CTkFrame(appearance_body, fg_color="transparent")
    ui_text_row.pack(fill="x", pady=(6, 4))
    ctk.CTkLabel(ui_text_row, text="App text size", font=FONT_UI, text_color=TEXT).grid(
        row=0, column=0, padx=(0, 8), sticky="w"
    )
    ctk.CTkSlider(
        ui_text_row,
        from_=0.9,
        to=1.7,
        number_of_steps=80,
        variable=text_scale_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).grid(row=0, column=1, sticky="we")
    ui_text_row.grid_columnconfigure(1, weight=1)

    chat_text_scale_var = tk.DoubleVar(
        value=prefs.get("chat_text_scale", prefs["text_scale"])
    )

    chat_text_row = ctk.CTkFrame(appearance_body, fg_color="transparent")
    chat_text_row.pack(fill="x", pady=(6, 4))
    ctk.CTkLabel(chat_text_row, text="Chat text size", font=FONT_UI, text_color=TEXT).grid(
        row=0, column=0, padx=(0, 8), sticky="w"
    )
    ctk.CTkSlider(
        chat_text_row,
        from_=0.9,
        to=1.7,
        number_of_steps=80,
        variable=chat_text_scale_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).grid(row=0, column=1, sticky="we")
    chat_text_row.grid_columnconfigure(1, weight=1)

    text_scale_labels = ctk.CTkFrame(appearance_body, fg_color="transparent")
    text_scale_labels.pack(fill="x", pady=(0, 10))
    for col, label in enumerate(["Smaller", "Default", "Larger"]):
        ctk.CTkLabel(
            text_scale_labels,
            text=label,
            font=("Segoe UI", 10),
            text_color=MUTED,
        ).grid(row=0, column=col, sticky="we")
        text_scale_labels.grid_columnconfigure(col, weight=1)
    temp_var = tk.DoubleVar(value=prefs["text_temperature"])
    max_tokens_var = tk.StringVar(value=str(prefs["text_max_tokens"]))

    text_body = _make_settings_card(
        scroll,
        "Text Responses",
        "Adjust how the chat responds. Lower temperatures stick to the facts, higher values let the model improvise. "
        "Token limits cap reply length (roughly 3–4 tokens per word).",
        title_font=font_h2,
        blurb_font=font_ui,
    )
    ctk.CTkLabel(text_body, text="Temperature", font=FONT_UI, text_color=TEXT).pack(
        anchor="w", pady=(0, 4)
    )
    ctk.CTkSlider(
        text_body,
        from_=0.0,
        to=1.5,
        number_of_steps=150,
        variable=temp_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).pack(fill="x", pady=(0, 8))

    ctk.CTkLabel(text_body, text="Max new tokens", font=FONT_UI, text_color=TEXT).pack(
        anchor="w", pady=(0, 4)
    )
    ctk.CTkEntry(text_body, textvariable=max_tokens_var).pack(fill="x", pady=(0, 12))

    width_var = tk.StringVar(value=str(prefs["image_width"]))
    height_var = tk.StringVar(value=str(prefs["image_height"]))
    steps_var = tk.IntVar(value=prefs["image_steps"])
    guidance_var = tk.DoubleVar(value=prefs["image_guidance"])
    seed_var = tk.StringVar(
        value="" if prefs["image_seed"] is None else str(prefs["image_seed"])
    )

    image_body = _make_settings_card(
        scroll,
        "Image Generation",
        "Choose default canvas size and diffusion behaviour. More steps and guidance make sharper images, "
        "but they take longer. Leave the seed blank for fresh randomness every run.",
        title_font=font_h2,
        blurb_font=font_ui,
    )
    for label, var in [("Width", width_var), ("Height", height_var)]:
        ctk.CTkLabel(image_body, text=label, font=font_ui, text_color=TEXT).pack(anchor="w", pady=(0, 4))
        ctk.CTkEntry(image_body, textvariable=var).pack(fill="x", pady=(0, 8))

    ctk.CTkLabel(image_body, text="Steps", font=font_ui, text_color=TEXT).pack(anchor="w", pady=(0, 4))
    ctk.CTkSlider(
        image_body,
        from_=1,
        to=50,
        number_of_steps=49,
        variable=steps_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).pack(fill="x", pady=(0, 8))

    ctk.CTkLabel(image_body, text="Guidance", font=font_ui, text_color=TEXT).pack(anchor="w", pady=(0, 4))
    ctk.CTkSlider(
        image_body,
        from_=0.0,
        to=7.5,
        number_of_steps=150,
        variable=guidance_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).pack(fill="x", pady=(0, 8))

    ctk.CTkLabel(image_body, text="Seed (blank = random)", font=font_ui, text_color=TEXT).pack(anchor="w", pady=(0, 4))
    ctk.CTkEntry(image_body, textvariable=seed_var).pack(fill="x", pady=(0, 8))

    device_var = tk.StringVar(value=prefs["device_preference"])
    device_values = ["cpu"] if is_jetson() else ["auto", "mps", "cuda", "cpu"]
    ui_scale_var = tk.DoubleVar(value=prefs["ui_scale"])
    history_var = tk.BooleanVar(value=prefs["history_enabled"])

    interface_body = _make_settings_card(
        scroll,
        "Interface & Performance",
        "Pick where heavy lifting happens and how big the UI renders. History keeps past chats per model "
        "for richer context—turn it off to keep sessions lighter.",
        title_font=font_h2,
        blurb_font=font_ui,
    )

    device_row = ctk.CTkFrame(interface_body, fg_color="transparent")
    device_row.pack(fill="x", pady=(0, 10))

    ctk.CTkLabel(device_row, text="Device preference", font=FONT_UI, text_color=TEXT).grid(
        row=0, column=0, padx=(0, 8), sticky="w"
    )

    ctk.CTkOptionMenu(
        device_row,
        values=device_values,
        variable=device_var,
        font=font_ui,
        dropdown_font=font_ui,
        fg_color=CONTROL_BG,
        text_color=TEXT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
        corner_radius=BUTTON_RADIUS,
    ).grid(row=0, column=1, sticky="w")

    ctk.CTkButton(
        interface_body,
        text="Run performance test (TinyLlama Q4_K_M)",
        command=_run_perf_test,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=font_ui,
        corner_radius=BUTTON_RADIUS,
    ).pack(anchor="w", pady=(4, 8), padx=4)


    scale_row = ctk.CTkFrame(interface_body, fg_color="transparent")
    scale_row.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(scale_row, text="UI scale", font=FONT_UI, text_color=TEXT).grid(
        row=0, column=0, padx=(0, 8), sticky="w"
    )
    ctk.CTkSlider(
        scale_row,
        from_=0.9,
        to=1.5,
        number_of_steps=60,
        variable=ui_scale_var,
        fg_color=PANEL_ELEVATED,
        progress_color=ACCENT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
    ).grid(row=0, column=1, sticky="we")
    scale_row.grid_columnconfigure(1, weight=1)

    ctk.CTkCheckBox(
        interface_body,
        text="Enable history (per-model, per-user)",
        variable=history_var,
        font=FONT_UI,
        text_color=TEXT,
        fg_color=ACCENT,
        border_color=ACCENT,
        hover_color=ACCENT_HOVER,
        checkmark_color=TEXT,
    ).pack(anchor="w", pady=(0, 4))

    ctk.CTkLabel(
        interface_body,
        text="Offline mode is always on so nothing leaves this machine.",
        font=font_ui,
        text_color=MUTED,
    ).pack(anchor="w", pady=(4, 0))

    tools_body = _make_settings_card(
        scroll,
        "Shortcuts",
        "Quick maintenance buttons. Use them to tidy files or free GPU memory without leaving the app.",
        title_font=font_h2,
        blurb_font=font_ui,
    )

    model_info_body = _make_settings_card(
        scroll,
        "Model Expansion (Requires a live internet connection)",
        (
            "The ACRE app is a fully offline operation. However, you can expand the model offering by adding "
            "your own additional LLMs with an internet connection.\n\n"
            "How it works:\n"
            "1. Visit https://huggingface.co/models.\n"
            "2. HuggingFace has a large resevoir of models available for use. Download a model of your choosing.\n"
            "3. If necessary, transfer the folder to this device via USB.\n"
            "4. Place the model folder inside the 'models' directory.\n"
            "5. Click 'Refresh' in the sidebar or restart the app.\n\n"
            "ACRE will automatically detect compatible models and list them for loading.\n\n"
            "Recommended models:\n"
            "• TinyLlama 1.1B — small, fast, good for general chat.\n"
            "• Mistral 7B — strong reasoning, very capable.\n"
            "• LLaMA 3 8B — excellent overall quality.\n"
            "• Qwen 3B/7B — great multilingual performance.\n"
            "• LLaVA or BakLLaVA — vision-capable models for image understanding.\n\n"
            "Tips:\n"
            "- Use GGUF models for best performance on CPU devices.\n"
            "- Larger models require more RAM. Be aware of overwhelming your hardware.\n"
            "- Ensure each model folder contains a proper config.json or gguf file.\n"
        ),
        title_font=font_h2,
        blurb_font=font_ui,
    )

    ctk.CTkButton(
        model_info_body,
        text="Open Models Folder",
        command=lambda: open_path(Path('models')),
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=font_bold,
    ).pack(anchor="w", pady=(8, 0))

    button_row = ctk.CTkFrame(tools_body, fg_color="transparent")
    button_row.pack(fill="x", pady=(0, 6))
    ctk.CTkButton(
        button_row,
        text="Open outputs",
        command=lambda: open_path(OUTPUTS_PATH / (gs.current_user or "")),
        font=font_bold,
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
    ).grid(row=0, column=0, padx=4, pady=6, sticky="we")
    ctk.CTkButton(
        button_row,
        text="Open models",
        command=lambda: open_path(Path("models")),
        font=font_bold,
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
    ).grid(row=0, column=1, padx=4, pady=6, sticky="we")
    ctk.CTkButton(
        button_row,
        text="Clear all histories",
        command=_clear_all_histories,
        corner_radius=BUTTON_RADIUS,
        fg_color=CRITICAL,
        hover_color=CRITICAL_HOVER,
        text_color="white",
        font=font_bold,
    ).grid(row=0, column=2, padx=4, pady=6, sticky="we")
    ctk.CTkButton(
        button_row,
        text="Free VRAM / Clear caches",
        command=_clear_caches,
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=font_bold,
    ).grid(row=0, column=3, padx=4, pady=6, sticky="we")
    for column in range(4):
        button_row.grid_columnconfigure(column, weight=1)

    diag_body = _make_settings_card(
        scroll,
        "Diagnostics",
        "Copies of library versions so you can verify what shipped with this build.",
        title_font=font_h2,
        blurb_font=font_ui,
    )
    try:
        import platform

        info = [f"Python: {platform.python_version()}"]
        for mod in (
            "torch",
            "transformers",
            "diffusers",
            "llama_cpp",
            "google.protobuf",
            "soundfile",
        ):
            try:
                module = __import__(mod)
                version = getattr(module, "__version__", "unknown")
                info.append(f"{mod}: {version}")
            except Exception:
                info.append(f"{mod}: not available")
        info.append(
            f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE','')} "
            f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE','')}"
        )
        diag_text = ctk.CTkTextbox(
            diag_body,
            corner_radius=RADIUS_SM,
            fg_color=SURFACE_PRIMARY,
            text_color=TEXT,
            font=font_ui,
            wrap="word",
            height=160,
        )
        diag_text.pack(fill="both", expand=True)
        diag_text.insert("1.0", "\n".join(info))
        diag_text.configure(state="disabled")
    except Exception:
        ctk.CTkLabel(
            diag_body,
            text="Unable to gather diagnostics on this platform.",
            font=font_ui,
            text_color=MUTED,
        ).pack(anchor="w")

    training_body = _make_settings_card(
        scroll,
        "Model Training",
        "Fine-tune a compatible base model on your own dataset. This opens the guided training setup and runs locally.",
        title_font=font_h2,
        blurb_font=font_ui,
    )
    ctk.CTkButton(
        training_body,
        text="Open training setup",
        command=open_training_dialog,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=font_bold,
        corner_radius=BUTTON_RADIUS,
    ).pack(anchor="w", pady=(4, 0))

    def save_settings_values() -> None:
        new_values = {
            "text_temperature": to_float(temp_var.get(), 0.7),
            "text_max_tokens": to_int(max_tokens_var.get(), 512),
            "image_width": to_int(width_var.get(), 512),
            "image_height": to_int(height_var.get(), 512),
            "image_steps": int(steps_var.get()),
            "image_guidance": to_float(guidance_var.get(), 0.0),
            "image_seed": (
                to_int(seed_var.get(), None) if str(seed_var.get()).strip() else None
            ),
            "ui_scale": to_float(ui_scale_var.get(), 1.15),
            "device_preference": str(device_var.get()).lower(),
            "history_enabled": bool(history_var.get()),
            "text_scale": to_float(text_scale_var.get(), 1.15),
            "chat_text_scale": to_float(chat_text_scale_var.get(), 1.15),
        }
        set_prefs(new_values)
        if gs.mgr:
            gs.mgr.set_history_enabled(new_values["history_enabled"])
            gs.mgr.set_text_config(
                max_tokens=new_values["text_max_tokens"],
                temperature=new_values["text_temperature"],
            )

        ui_scale_value = float(new_values["ui_scale"])
        text_scale_value = float(new_values.get("text_scale", ui_scale_value))
        chat_scale_value = float(
            new_values.get("chat_text_scale", text_scale_value)
        )

        try:
            ctk.set_widget_scaling(ui_scale_value)
            apply_native_font_scale(text_scale_value)

            family, base_size = FONT_UI[0], FONT_UI[1]
            rest = FONT_UI[2:] if len(FONT_UI) > 2 else ()
            new_size = max(8, int(base_size * chat_scale_value))
            chat_font = (family, new_size) + rest

            if gs.chat_history:
                gs.chat_history.configure(font=chat_font)
            if gs.entry:
                gs.entry.configure(font=chat_font)
        except Exception:
            pass

        update_status("Settings saved.")
        render_settings_tab(tab)
        refresh_main_ui()

    ctk.CTkButton(
        scroll,
        text="Save changes",
        command=save_settings_values,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=font_bold,
        corner_radius=BUTTON_RADIUS,
    ).pack(pady=(16, 20))
    ctk.CTkLabel(
        scroll,
        text="Made with <3 at the University of Missouri",
        font=font_ui,
        text_color=MUTED,
    ).pack(pady=(0, 16))

def open_settings() -> None:
    if not gs.tabs:
        return
    try:
        tab = gs.tabs.tab("Settings")
    except Exception:
        tab = getattr(gs, "settings_tab", None)
    if tab is None:
        return
    render_settings_tab(tab)
    try:
        gs.tabs.set("Settings")
    except Exception:
        pass


def build_main_ui() -> None:
    _refresh_theme_vars()
    settings = load_settings()
    history_dir = str(Path("history") / gs.current_user) if gs.current_user else "history"
    prefs = get_prefs()
    
    chat_scale = prefs.get(
        "chat_text_scale",
        prefs.get("text_scale", prefs["ui_scale"]),
    )
    family, base_size = FONT_UI[0], FONT_UI[1]
    rest = FONT_UI[2:] if len(FONT_UI) > 2 else ()
    chat_font = (family, max(8, int(base_size * chat_scale))) + rest

    from model_manager import ModelManager

    if gs.mgr is None:
        gs.mgr = ModelManager(
            models_dir=str(Path("models")),
            history_dir=history_dir,
            device_pref=prefs["device_preference"],
        )
    if gs.encryption_key:
        try:
            encryptor = ChatEncryptor(gs.encryption_key)
            gs.mgr.set_encryptor(encryptor)
        except Exception:
            update_status("Encryption disabled: unable to initialize secure storage.")
    gs.mgr.set_history_enabled(prefs["history_enabled"])
    gs.mgr.set_text_config(
        max_tokens=prefs["text_max_tokens"], temperature=prefs["text_temperature"]
    )
    side_pack = dict(side="left", fill="y", padx=(10, 8), pady=12)
    gs.side_frame = ctk.CTkFrame(
        gs.workspace_frame,
        width=260,
        fg_color=PANEL_ELEVATED,
        corner_radius=RADIUS_LG,
        border_width=0,
    )
    gs.side_frame.pack(**side_pack)
    gs.side_frame.pack_propagate(False)
    header = ctk.CTkFrame(gs.side_frame, fg_color="transparent")
    header.pack(fill="x", pady=(12, 6), padx=8)
    models_label = ctk.CTkLabel(
        header,
        text="Models",
        text_color=TEXT,
        font=FONT_H2,
    )
    models_label.pack(anchor="w", padx=4, pady=4)
    listbox_shell = ctk.CTkFrame(
        gs.side_frame,
        fg_color=GLASS_BG,
        corner_radius=RADIUS_LG,
        border_width=0,
    )
    listbox_shell.pack(fill="both", expand=True, padx=6, pady=6)
    listbox_frame = ctk.CTkFrame(
        listbox_shell,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_LG,
        border_width=0,
    )
    listbox_frame.pack(fill="both", expand=True, padx=8, pady=8)
    gs.listbox = tk.Listbox(
        listbox_frame,
        bg=CONTROL_BG,
        fg=TEXT,
        selectbackground=ACCENT,
        selectforeground=TEXT,
        relief=tk.FLAT,
        highlightthickness=0,
        bd=0,
    )
    gs.listbox.pack(fill="both", expand=True, padx=0, pady=2)
    apply_native_font_scale(prefs.get("text_scale", prefs["ui_scale"]))
    menu = tk.Menu(gs.listbox, tearoff=0, font=FONT_UI)
    menu.add_command(label="Rename…", command=rename_model)

    def reveal_selected() -> None:
        selection = gs.listbox.get(tk.ACTIVE)
        if not selection:
            return
        real_name = gs.alias_to_real.get(selection, selection)
        path = Path("models") / real_name
        if not path.exists():
            return
        open_path(path)

    menu.add_command(label="Reveal in Finder", command=reveal_selected)

    def show_menu(event) -> None:
        try:
            index = gs.listbox.nearest(event.y)
            gs.listbox.selection_clear(0, tk.END)
            gs.listbox.selection_set(index)
            gs.listbox.activate(index)
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    gs.listbox.bind("<Button-3>", show_menu)
    gs.listbox.bind("<Control-Button-1>", show_menu)

    button_kwargs = dict(fill="x", pady=5, padx=10)
    primary_button = dict(
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
    )
    ctk.CTkButton(
        gs.side_frame,
        text="Refresh",
        command=refresh_list,
        **primary_button,
    ).pack(**button_kwargs)
    ctk.CTkButton(
        gs.side_frame,
        text="Load Model",
        command=pick_model,
        **primary_button,
    ).pack(**button_kwargs)
    ctk.CTkButton(
        gs.side_frame,
        text="Add Model",
        command=add_model,
        **primary_button,
    ).pack(**button_kwargs)
    ctk.CTkButton(
        gs.side_frame,
        text="Clear History",
        command=clear_chat,
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
    ).pack(**button_kwargs)
    ctk.CTkButton(
        gs.side_frame,
        text="Switch User",
        command=logout_action,
        corner_radius=BUTTON_RADIUS,
        fg_color=CRITICAL,
        hover_color=CRITICAL_HOVER,
        text_color="white",
        font=FONT_BOLD,
    ).pack(**button_kwargs)
    gs.user_label = ctk.CTkLabel(
        gs.side_frame,
        text=f"User: {gs.current_user or '-'}",
        text_color=MUTED,
        font=FONT_UI,
        anchor="w",
    )
    gs.user_label.pack(pady=(6, 2), padx=4, fill="x")
    status_holder = ctk.CTkFrame(
        gs.side_frame,
        fg_color=GLASS_BG,
        corner_radius=RADIUS_MD,
        border_width=0,
    )
    status_holder.pack(fill="x", padx=4, pady=(0, 8))
    gs.status = ctk.CTkTextbox(
        status_holder,
        height=84,
        corner_radius=RADIUS_MD,
        fg_color=SURFACE_PRIMARY,
        text_color=MUTED,
        font=FONT_UI,
        wrap="word",
    )
    gs.status.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    gs.status_scroll = ctk.CTkScrollbar(status_holder, command=gs.status.yview)
    gs.status_scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)
    gs.status.configure(yscrollcommand=gs.status_scroll.set, state="disabled")
    right = ctk.CTkFrame(
        gs.workspace_frame,
        fg_color="transparent",
        corner_radius=0,
        border_width=0,
    )
    right.pack(side="right", fill="both", expand=True, padx=(6, 10), pady=10)
    gs.chat_frame = right
    gs.tabs = ctk.CTkTabview(right, fg_color="transparent", border_width=0, corner_radius=RADIUS_LG)
    gs.tabs.pack(fill="both", expand=True, padx=8, pady=8)
    chat_tab = gs.tabs.add("Chat")
    gallery_tab = gs.tabs.add("Gallery")
    settings_tab = gs.tabs.add("Settings")
    gs.settings_tab = settings_tab
    try:
        gs.tabs._segmented_button.configure(
            font=FONT_UI,
            fg_color=GLASS_BG,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_color=PANEL_ELEVATED,
            unselected_hover_color=GLASS_BG,
            selected_text_color=TEXT,
            unselected_text_color=TEXT,
        )       
    except Exception:
        pass
    render_settings_tab(settings_tab)
    chat_border = ctk.CTkFrame(
        chat_tab,
        fg_color="transparent",
        corner_radius=0,
        border_width=0,
    )
    chat_border.pack(fill="both", expand=True, padx=12, pady=(12, 8))
    message_panel = ctk.CTkFrame(
        chat_border,
        fg_color=GLASS_BG,
        corner_radius=RADIUS_LG,
    )
    message_panel.pack(fill="both", expand=True)
    gs.chat_history = ctk.CTkTextbox(
        message_panel,
        height=460,
        corner_radius=RADIUS_LG,
        fg_color=SURFACE_PRIMARY,
        text_color=TEXT,
        font=chat_font,
        wrap="word",
    )
    gs.chat_history.pack(fill="both", expand=True, padx=18, pady=18)
    gs.chat_history.configure(state="disabled")
    gs.logo_label = None
    try:
        logo_path = Path("transparent-logo.png")
        if logo_path.exists():
            with Image.open(logo_path) as logo_image:
                logo_resized = logo_image.resize((220, 220))
                logo_tk = ImageTk.PhotoImage(logo_resized)
            gs.logo_label = tk.Label(message_panel, image=logo_tk, bg=SURFACE_PRIMARY)
            gs.logo_label.image = logo_tk
            gs.logo_label.place(relx=0.5, rely=0.5, anchor="center")
            gs.logo_label.lift()
    except Exception:
        gs.logo_label = None
    gs.attach_row = ctk.CTkFrame(
        chat_tab,
        fg_color=GLASS_BG,
        corner_radius=RADIUS_MD,
        border_width=0,
    )
    gs.attach_row.pack(fill="x", padx=12, pady=(8, 10))
    refresh_attach_row()
    entry_border = ctk.CTkFrame(
        chat_tab,
        fg_color=GLASS_BG,
        corner_radius=RADIUS_LG,
        border_width=0,
    )
    entry_border.pack(fill="x", padx=12, pady=(0, 16))
    entry_container = ctk.CTkFrame(
        entry_border,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_MD,
    )
    entry_container.pack(fill="x", padx=12, pady=12)
    send_button = ctk.CTkButton(
        entry_container,
        text="↑",
        width=52,
        height=42,
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=run_prompt,
    )
    send_button.pack(side="right", padx=(6, 8), pady=4)
    entry_frame = ctk.CTkFrame(
        entry_container,
        fg_color=CONTROL_BG,
        corner_radius=RADIUS_MD,
    )
    entry_frame.pack(side="left", fill="both", expand=True, padx=(8, 6), pady=4)
    gs.entry = tk.Text(
        entry_frame,
        height=1,
        wrap="word",
        bg=CONTROL_BG,
        fg=MUTED,
        insertbackground=TEXT,
        relief="flat",
        highlightthickness=0,
        bd=0,
        font=chat_font,
    )
    gs.entry.pack(fill="both", expand=True, padx=6, pady=4)
    gs.entry.insert("1.0", "Ask me anything...")
    apply_native_font_scale(prefs.get("text_scale", prefs["ui_scale"]))

    def on_focus_in(event) -> None:
        if gs.entry.get("1.0", "end-1c") == "Ask me anything...":
            gs.entry.delete("1.0", "end")
            gs.entry.configure(fg=TEXT)

    def on_focus_out(event) -> None:
        if gs.entry.get("1.0", "end-1c").strip() == "":
            gs.entry.insert("1.0", "Ask me anything...")
            gs.entry.configure(fg=MUTED)

    gs.entry.bind("<FocusIn>", on_focus_in)
    gs.entry.bind("<FocusOut>", on_focus_out)

    def auto_resize(event=None) -> None:
        lines = int(gs.entry.index("end-1c").split(".")[0])
        gs.entry.configure(height=min(max(1, lines), 8))

    gs.entry.bind("<KeyRelease>", auto_resize)

    def send_on_return(event):
        if event.state & 0x0001:
            return
        run_prompt()
        return "break"

    gs.entry.bind("<Return>", send_on_return)
    gs.entry.bind("<KP_Enter>", send_on_return)
    gallery_top = ctk.CTkFrame(gallery_tab, fg_color="transparent")
    gallery_top.pack(fill="x", padx=12, pady=(12, 0))
    ctk.CTkLabel(gallery_top, text="Recent images", font=FONT_H2, text_color=TEXT).pack(
        side="left", padx=4, pady=6
    )
    ctk.CTkButton(
        gallery_top,
        text="Open outputs",
        command=lambda: open_path(OUTPUTS_PATH / (gs.current_user or "")),
        font=FONT_BOLD,
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
    ).pack(side="right", padx=10, pady=8)
    gallery_body = ctk.CTkScrollableFrame(
        gallery_tab,
        fg_color="transparent",
        orientation="horizontal",
    )
    gallery_body.pack(fill="both", expand=True, padx=12, pady=12)
    gs.gallery_container = gallery_body
    refresh_gallery(gs.gallery_container)
    toggle_state = {"open": True}
    gs.toggle_btn = None

    def show_reopen_button() -> None:
        if gs.toggle_btn:
            return
        gs.toggle_btn = ctk.CTkButton(
            gs.workspace_frame,
            text="▶ Models",
            width=118,
            height=36,
            corner_radius=BUTTON_RADIUS,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="white",
            font=FONT_BOLD,
            command=toggle_sidebar,
        )
        gs.toggle_btn.place(x=12, y=20)
        gs.toggle_btn.lift()

    def hide_reopen_button() -> None:
        if gs.toggle_btn:
            gs.toggle_btn.destroy()
            gs.toggle_btn = None

    def toggle_sidebar(_event=None) -> None:
        if toggle_state["open"]:
            hide_reopen_button()
            gs.side_frame.pack_forget()
            toggle_state["open"] = False
            show_reopen_button()
        else:
            hide_reopen_button()
            gs.side_frame.pack(before=right, **side_pack)
            toggle_state["open"] = True
        if gs.root:
            gs.root.after(50, lambda: None)

    def header_click(_event=None) -> None:
        toggle_sidebar()

    header.bind("<Button-1>", header_click)
    models_label.bind("<Button-1>", header_click)
    toggle_state["open"] = True

    gs.settings_btn = ctk.CTkButton(
        gs.workspace_frame,
        text="⚙",
        width=40,
        height=36,
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=open_settings,
    )
    gs.settings_btn.place(relx=1.0, x=-24, y=26, anchor="ne")
    gs.settings_btn.lift()
    refresh_list()
    render_history()
    if gs.mgr and gs.mgr.is_loaded():
        try:
            update_status(gs.mgr.describe_session())
        except Exception:
            update_status("Model loaded")
    else:
        update_status("No model loaded")


def refresh_main_ui() -> None:
    # Rebuild the main UI to apply theme/font changes without touching the model manager.
    if gs.workspace_frame is None:
        return
    _refresh_theme_vars()
    teardown_main_ui()
    build_main_ui()
    try:
        if gs.root:
            gs.root.configure(fg_color=BG_GRAD_TOP)
            recolor_whole_app(gs.root)
    except Exception:
        pass


def show_frame(frame) -> None:
    frame.lift()
    if frame is gs.login_frame and gs.login_password_entry:
        try:
            gs.root.after(50, lambda: gs.login_password_entry.focus_force())
        except Exception:
            pass
    elif frame is gs.setup_frame:
        try:
            gs.root.after(50, lambda: gs.gate_frame.focus_force() if gs.gate_frame else None)
        except Exception:
            pass


def build_gate_ui() -> None:
    if not gs.workspace_frame:
        return
    settings = load_settings()
    users = list_usernames(settings)
    first_run = len(users) == 0
    remembered_info = get_remembered_user(settings)
    remember_ready: Optional[tuple[str, bytes]] = None
    remember_pending: Optional[tuple[str, bytes]] = None
    if remembered_info:
        remembered_user, remembered_key = remembered_info
        record = ensure_users_bucket(settings).get(remembered_user, {})
        if record and record.get("disclaimer_ack"):
            remember_ready = (remembered_user, remembered_key)
        else:
            remember_pending = (remembered_user, remembered_key)
    gs.login_password_entry = None
    gs.pending_user = None
    gs.encryption_key = None
    gs.gate_frame = ctk.CTkFrame(
        gs.workspace_frame,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_LG,
        border_width=1,
        border_color=CONTROL_BORDER,
    )
    gs.gate_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.92, relheight=0.92)
    gs.gate_frame.lift()
    try:
        gs.gate_frame.focus_set()
    except Exception:
        pass
    if gs.root:
        gs.root.after(50, lambda: gs.root.focus_force())
    gs.setup_frame = ctk.CTkFrame(
        gs.gate_frame,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_LG,
    )
    gs.login_frame = ctk.CTkFrame(
        gs.gate_frame,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_LG,
    )
    gs.disc_frame = ctk.CTkFrame(
        gs.gate_frame,
        fg_color=SURFACE_PRIMARY,
        corner_radius=RADIUS_LG,
    )
    for frame in (gs.setup_frame, gs.login_frame, gs.disc_frame):
        frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.96, relheight=0.96)
    ctk.CTkLabel(gs.setup_frame, text="Create Account", font=FONT_H1, text_color=TEXT).pack(
        pady=(18, 10)
    )
    wrap = ctk.CTkFrame(
        gs.setup_frame,
        fg_color=SURFACE_ELEVATED,
        corner_radius=RADIUS_MD,
        border_width=1,
        border_color=CONTROL_BORDER,
    )
    wrap.pack(pady=12, padx=12)
    wrap.grid_columnconfigure(1, weight=1)
    username_var = tk.StringVar(value="")
    password_var = tk.StringVar(value="")
    confirm_var = tk.StringVar(value="")
    row_kwargs = dict(padx=8, pady=8, sticky="we")
    ctk.CTkLabel(wrap, text="Username", text_color=TEXT, font=FONT_UI).grid(
        row=0, column=0, sticky="w", padx=8, pady=8
    )
    ctk.CTkEntry(wrap, textvariable=username_var, font=FONT_UI).grid(
        row=0, column=1, **row_kwargs
    )
    ctk.CTkLabel(wrap, text="Password", text_color=TEXT, font=FONT_UI).grid(
        row=1, column=0, sticky="w", padx=8, pady=8
    )
    ctk.CTkEntry(wrap, textvariable=password_var, show="*", font=FONT_UI).grid(
        row=1, column=1, **row_kwargs
    )
    ctk.CTkLabel(wrap, text="Confirm", text_color=TEXT, font=FONT_UI).grid(
        row=2, column=0, sticky="w", padx=8, pady=8
    )
    ctk.CTkEntry(wrap, textvariable=confirm_var, show="*", font=FONT_UI).grid(
        row=2, column=1, **row_kwargs
    )

    def handle_setup() -> None:
        settings_local = load_settings()
        users_local = ensure_users_bucket(settings_local)
        name = username_var.get().strip()
        password_value = password_var.get().strip()
        confirm_value = confirm_var.get().strip()
        if not name or name in users_local:
            messagebox.showerror("Setup Failed", "Invalid or duplicate username.")
            return
        if password_value != confirm_value:
            messagebox.showerror("Setup Failed", "Passwords do not match.")
            return
        try:
            record = set_credentials(password_value)
        except ValueError as exc:
            messagebox.showerror("Setup Failed", str(exc))
            return
        try:
            key = derive_fernet_key(
                password_value,
                salt_b64=record["enc_salt"],
                iterations=record.get("enc_iterations", record.get("iterations", 200_000)),
            )
        except Exception as exc:
            messagebox.showerror("Setup Failed", f"Could not initialize encryption: {exc}")
            return
        users_local[name] = record
        settings_local["active_user"] = name
        save_settings(settings_local)
        gs.pending_user = name
        gs.encryption_key = key
        username_var.set("")
        password_var.set("")
        confirm_var.set("")
        show_frame(gs.disc_frame)

    ctk.CTkButton(
        gs.setup_frame,
        text="Save & Continue",
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=handle_setup,
    ).pack(pady=12)
    ctk.CTkLabel(
        gs.login_frame, text="ACRE LLM Switchboard", font=FONT_H1, text_color=TEXT
    ).pack(pady=(24, 10))
    form = ctk.CTkFrame(
        gs.login_frame,
        fg_color=SURFACE_ELEVATED,
        corner_radius=RADIUS_MD,
        border_width=1,
        border_color=CONTROL_BORDER,
    )
    form.pack(pady=12, padx=12)
    form.grid_columnconfigure(1, weight=1)
    users_list = users or ["admin"]
    preferred_user = get_active_user(settings)
    if not preferred_user or preferred_user not in users_list:
        if remembered_info and remembered_info[0] in users_list:
            preferred_user = remembered_info[0]
        else:
            preferred_user = users_list[0]
    who_var = tk.StringVar(value=preferred_user)
    password_login = tk.StringVar(value="")
    def has_valid_remember(user: str) -> bool:
        record = ensure_users_bucket(settings).get(user, {})
        if not isinstance(record, dict):
            return False
        try:
            expiry_value = float(record.get("remember_expires", 0))
        except Exception:
            expiry_value = 0.0
        return bool(record.get("remember_key") and expiry_value > time.time())

    remember_var = tk.BooleanVar(value=has_valid_remember(preferred_user))
    def on_user_select(*_args) -> None:
        remember_var.set(has_valid_remember(who_var.get().strip()))

    who_var.trace_add("write", on_user_select)
    ctk.CTkLabel(form, text="User", text_color=TEXT, font=FONT_UI).grid(
        row=0, column=0, padx=8, pady=8, sticky="w"
    )
    option_menu = ctk.CTkOptionMenu(
        form,
        values=users_list,
        variable=who_var,
        font=FONT_UI,
        dropdown_font=FONT_UI,
        fg_color=CONTROL_BG,
        text_color=TEXT,
        button_color=ACCENT,
        button_hover_color=ACCENT_HOVER,
        corner_radius=BUTTON_RADIUS,
    )
    option_menu.grid(row=0, column=1, padx=8, pady=8, sticky="we")

    option_menu.grid(row=0, column=1, padx=8, pady=8, sticky="we")
    ctk.CTkLabel(form, text="Password", text_color=TEXT, font=FONT_UI).grid(
        row=1, column=0, padx=8, pady=8, sticky="w"
    )
    password_entry = ctk.CTkEntry(
        form, textvariable=password_login, show="*", font=FONT_UI
    )
    password_entry.grid(row=1, column=1, padx=8, pady=8, sticky="we")
    gs.login_password_entry = password_entry
    remember_box = ctk.CTkCheckBox(
        form,
        text="Remember me on this device for 30 days",
        variable=remember_var,
        text_color=TEXT,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        font=FONT_UI,
    )
    remember_box.grid(row=2, column=0, columnspan=2, padx=8, pady=(0, 4), sticky="w")

    def _focus_login_password() -> None:
        if password_entry.winfo_exists():
            password_entry.focus_force()

    if gs.root:
        gs.root.after(100, _focus_login_password)

    def handle_login() -> None:
        settings_local = load_settings()
        username = who_var.get().strip()
        password_value = password_login.get().strip()
        users_local = ensure_users_bucket(settings_local)
        if username not in users_local:
            messagebox.showerror("Login Failed", "Unknown user.")
            return
        if not verify_password(username, password_value, settings_local):
            messagebox.showerror("Login Failed", "Invalid credentials.")
            password_login.set("")
            password_entry.focus_set()
            return
        record = ensure_encryption_metadata(
            settings_local, username, default_iterations=users_local[username].get("iterations")
        )
        if record is None:
            messagebox.showerror("Login Failed", "Account data is missing or invalid.")
            password_login.set("")
            return
        try:
            key = derive_fernet_key(
                password_value,
                salt_b64=record["enc_salt"],
                iterations=record.get("enc_iterations", record.get("iterations", 200_000)),
            )
        except Exception as exc:
            messagebox.showerror("Login Failed", f"Could not derive encryption key: {exc}")
            password_login.set("")
            return
        gs.encryption_key = key
        gs.pending_user = username
        if remember_var.get():
            expires_at = time.time() + 30 * 24 * 60 * 60
            set_remember_me(settings_local, username, key, expires_at)
        else:
            clear_remember_me(settings_local, username)
        set_active_user(settings_local, username)
        password_login.set("")
        if bool(record.get("disclaimer_ack")):
            complete_login(mark_ack=False)
        else:
            show_frame(gs.disc_frame)

    row = ctk.CTkFrame(gs.login_frame, fg_color=SURFACE_PRIMARY)
    row.pack(pady=12)
    ctk.CTkButton(
        row,
        text="Login",
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=handle_login,
    ).grid(row=0, column=0, padx=6)
    ctk.CTkButton(
        row,
        text="Create Account",
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=lambda: show_frame(gs.setup_frame),
    ).grid(row=0, column=1, padx=6)

    def complete_login(*, mark_ack: bool) -> None:
        settings_local = load_settings()
        user = gs.pending_user or get_active_user(settings_local)
        if not user:
            messagebox.showerror("Login Failed", "No active user selected.")
            return
        if mark_ack:
            set_disclaimer_ack(settings_local, user, True)
        else:
            ensure_encryption_metadata(settings_local, user)
        gs.current_user = user
        gs.pending_user = None
        try:
            Path("models").mkdir(parents=True, exist_ok=True)
            Path("history").mkdir(parents=True, exist_ok=True)
            Path("outputs").mkdir(parents=True, exist_ok=True)
            ensure_user_dirs()
        except PermissionError:
            messagebox.showerror(
                "Permission Denied",
                "Unable to create required directories. Please check file permissions."
            )
            return
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to create directories: {e}"
            )
            return
        try:
            if gs.gate_frame:
                gs.gate_frame.destroy()
        except Exception:
            pass
        gs.gate_frame = None
        gs.setup_frame = None
        gs.login_frame = None
        gs.disc_frame = None
        gs.login_password_entry = None
        build_main_ui()

    ctk.CTkLabel(
        gs.disc_frame, text="IMPORTANT DISCLAIMER", font=FONT_H1, text_color="#ffcccc"
    ).pack(pady=(18, 8))
    wrapper = ctk.CTkFrame(
        gs.disc_frame,
        fg_color=SURFACE_ELEVATED,
        corner_radius=RADIUS_MD,
        border_width=1,
        border_color=CONTROL_BORDER,
    )
    wrapper.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    scroll = ctk.CTkScrollbar(wrapper)
    scroll.pack(side="right", fill="y", padx=(0, 6), pady=8)
    text_box = ctk.CTkTextbox(
        wrapper,
        corner_radius=RADIUS_MD,
        fg_color=CONTROL_BG,
        text_color=TEXT,
        font=FONT_UI,
        wrap="word",
    )
    text_box.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=8)
    from .constants import DISCLAIMER_TEXT

    text_box.insert("1.0", DISCLAIMER_TEXT)
    text_box.configure(state="disabled")
    text_box.configure(yscrollcommand=scroll.set)
    scroll.configure(command=text_box.yview)
    ack_var = tk.BooleanVar(value=False)

    def on_check() -> None:
        ok_button.configure(state="normal" if ack_var.get() else "disabled")

    ctk.CTkCheckBox(
        gs.disc_frame,
        text="I have read and understand the disclaimer",
        variable=ack_var,
        command=on_check,
        text_color=TEXT,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        font=FONT_UI,
    ).pack(pady=10)

    def accept() -> None:
        complete_login(mark_ack=True)

    ok_button = ctk.CTkButton(
        gs.disc_frame,
        text="I Understand",
        state="disabled",
        corner_radius=BUTTON_RADIUS,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=accept,
    )
    ok_button.pack(pady=(0, 8))

    def back_to_auth() -> None:
        gs.pending_user = None
        gs.encryption_key = None
        ack_var.set(False)
        ok_button.configure(state="disabled")
        destination = gs.login_frame if list_usernames(load_settings()) else gs.setup_frame
        show_frame(destination)

    ctk.CTkButton(
        gs.disc_frame,
        text="Back",
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
        command=back_to_auth,
    ).pack(pady=(0, 12))

    if remember_ready:
        set_active_user(settings, remember_ready[0])
        gs.pending_user = remember_ready[0]
        gs.encryption_key = remember_ready[1]
        complete_login(mark_ack=False)
        return

    target_frame = gs.setup_frame if first_run else gs.login_frame
    if remember_pending:
        set_active_user(settings, remember_pending[0])
        gs.pending_user = remember_pending[0]
        gs.encryption_key = remember_pending[1]
        target_frame = gs.disc_frame
    show_frame(target_frame)


def on_close() -> None:
    try:
        if gs.mgr:
            gs.mgr.unload()
    finally:
        gs.root.destroy()


def _check_display_server() -> tuple[bool, Optional[str]]:
    import os
    import sys
    import subprocess
    import shutil

    if sys.platform == "darwin" or sys.platform == "win32":
        return True, None

    def _display_is_available() -> bool:
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                timeout=2,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except (FileNotFoundError, OSError):
            # If xdpyinfo is missing we assume the display might still be fine.
            return True

    if os.environ.get("DISPLAY") and _display_is_available():
        return True, None

    xvfb_path = shutil.which("Xvfb")
    if xvfb_path:
        target_display = ":99"
        try:
            subprocess.Popen(
                [xvfb_path, target_display, "-screen", "0", "1024x768x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.3)
            os.environ["DISPLAY"] = target_display
        except OSError as exc:
            return False, f"Failed to start Xvfb automatically: {exc}. GUI requires X11 display server."

        if _display_is_available():
            return True, None
        return False, "Attempted to start Xvfb automatically, but X server is still not accessible."

    if not os.environ.get("DISPLAY"):
        return False, "DISPLAY environment variable is not set and Xvfb is not installed. GUI requires X11 display server."

    return False, "X server is not accessible. Please start X server or install Xvfb for headless mode."


def run_app() -> None:
    display_ok, display_error = _check_display_server()
    if not display_ok:
        import sys
        print(f"ERROR: {display_error}", file=sys.stderr)
        print("TIP: For headless Linux systems, use Xvfb:", file=sys.stderr)
        print("  Xvfb :99 -screen 0 1024x768x24 &", file=sys.stderr)
        print("  export DISPLAY=:99", file=sys.stderr)
        sys.exit(1)
    
    prefs = get_prefs()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    gs.root = ctk.CTk()
    gs.root.geometry("1000x800")
    gs.root.configure(fg_color=BG_GRAD_TOP)
    gs.workspace_frame = ctk.CTkFrame(
        gs.root,
        fg_color="transparent",
        corner_radius=0,
        border_width=0,
    )
    gs.workspace_frame.pack(fill="both", expand=True, padx=8, pady=8)
    ctk.set_widget_scaling(prefs["ui_scale"])
    apply_native_font_scale(prefs.get("text_scale", prefs["ui_scale"]))
    build_gate_ui()
    gs.root.protocol("WM_DELETE_WINDOW", on_close)
    gs.root.mainloop()
