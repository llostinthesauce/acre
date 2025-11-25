import os
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from . import global_state as gs
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BUTTON_RADIUS,
    FONT_BOLD,
    FONT_UI,
    MODELS_PATH,
    SURFACE_ELEVATED,
    TEXT,
    TITLE_BAR_ACCENT,
)
from .settings import (
    get_alias_map,
    get_prefs,
    load_settings,
    set_alias,
)
from .ui_helpers import update_status


def rebuild_model_listbox() -> None:
    if not gs.mgr or not gs.listbox:
        return
    settings = load_settings()
    aliases = get_alias_map(settings, gs.current_user or "_global_")
    models = gs.mgr.list_models()
    gs.listbox.delete(0, tk.END)
    gs.alias_to_real.clear()
    gs.real_to_alias.clear()
    for real_name in models:
        display = aliases.get(real_name, real_name)
        gs.alias_to_real[display] = real_name
        gs.real_to_alias[real_name] = display
        gs.listbox.insert(tk.END, display)


def refresh_list() -> None:
    rebuild_model_listbox()


def pick_model() -> None:
    if not gs.mgr or not gs.listbox:
        return
    choice = gs.listbox.get(tk.ACTIVE)
    if not choice:
        update_status("Select a model to load.")
        return
    real_name = gs.alias_to_real.get(choice, choice)
    prefs = get_prefs()
    loaded, message = gs.mgr.load_model(real_name, device_pref=prefs["device_preference"])
    if loaded:
        gs.mgr.set_history_enabled(prefs["history_enabled"])
        gs.mgr.set_text_config(
            max_tokens=prefs["text_max_tokens"], temperature=prefs["text_temperature"]
        )
        from .chat import render_history
        from .attachments import refresh_attach_row

        render_history()
        summary = gs.mgr.describe_session()
        update_status(summary)
        refresh_attach_row()
    else:
        detail = f" — {message}" if message else ""
        update_status(f"Failed to load: {choice}{detail}")


def add_model() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    folder = messagebox.askyesno(
        "Import Model", "Import a model FOLDER? (Yes = Folder, No = Single File)"
    )
    if folder:
        directory = filedialog.askdirectory(title="Pick a model folder")
        if not directory:
            return
        name = os.path.basename(directory.rstrip("/"))
        destination = MODELS_PATH / name
        if destination.exists():
            messagebox.showerror("Already Exists", f"{destination} already exists.")
            return
        try:
            shutil.copytree(directory, destination)
        except Exception as exc:
            update_status(f"Failed to add folder: {exc}")
            return
        refresh_list()
        update_status(f"Added model folder: {destination.name}")
        return
    file_path = filedialog.askopenfilename(
        title="Pick a model file",
        filetypes=[("Model files", "*.gguf"), ("All files", "*.*")],
    )
    if not file_path:
        return
    destination = MODELS_PATH / Path(file_path).name
    try:
        shutil.copy(file_path, destination)
    except Exception as exc:
        update_status(f"Failed to add file: {exc}")
        return
    refresh_list()
    update_status(f"Added model file: {destination.name}")


def rename_model() -> None:
    if not gs.listbox:
        return
    selected = gs.listbox.get(tk.ACTIVE)
    if not selected:
        return
    real_name = gs.alias_to_real.get(selected, selected)
    window = ctk.CTkToplevel(gs.root)
    window.title("Rename Model")
    window.resizable(False, False)
    ctk.CTkLabel(window, text="Display name", font=FONT_UI, text_color=TEXT).pack(padx=12, pady=(12, 6))
    value = tk.StringVar(value=gs.real_to_alias.get(real_name, real_name))
    entry = ctk.CTkEntry(window, textvariable=value, width=340, font=FONT_UI)
    entry.pack(padx=12, pady=6)

    def save_alias() -> None:
        settings = load_settings()
        user = gs.current_user or "_global_"
        new_value = value.get().strip()
        set_alias(settings, user, real_name, new_value if new_value and new_value != real_name else None)
        refresh_list()
        window.destroy()

    row = ctk.CTkFrame(window, fg_color=SURFACE_ELEVATED, corner_radius=BUTTON_RADIUS)
    row.pack(pady=10)
    ctk.CTkButton(
        row,
        text="Save",
        command=save_alias,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        corner_radius=BUTTON_RADIUS,
        text_color="white",
        font=FONT_BOLD,
    ).grid(row=0, column=0, padx=6, pady=8)
    ctk.CTkButton(
        row,
        text="Cancel",
        command=window.destroy,
        corner_radius=BUTTON_RADIUS,
        fg_color=TITLE_BAR_ACCENT,
        hover_color=ACCENT_HOVER,
        text_color="white",
        font=FONT_BOLD,
    ).grid(row=0, column=1, padx=6, pady=8)
    entry.focus_set()
