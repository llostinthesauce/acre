#!/usr/bin/env python3
"""Merged and cleaned app.py: customtkinter UI + ModelManager logic.

This file resolves the previous merge conflict. It prefers the
`customtkinter`-based UI but is defensive about UI updates so the app
doesn't crash during startup.
"""

import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk

from model_manager import ModelManager

# Theme and Fonts
BG_GRAD_TOP = "#0a1022"
BG_PANEL = "#0e1730"
BG_LIST = "#0b1328"
HL_LIST = "#1c2c56"
TEXT = "#e9f1ff"
MUTED = "#a9b8d6"
ACCENT = "#6ea5ff"
BORDER_ACCENT = "#91bbff"
FONT_UI = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_H1 = ("Segoe UI", 16, "bold")


mgr = ModelManager()


def update_status(message: str) -> None:
    try:
        status.configure(text=message)
    except Exception:
        # status might not be ready yet
        pass


def render_history() -> None:
    try:
        chat_history.configure(state='normal')
        chat_history.delete('1.0', tk.END)
        if mgr.is_loaded():
            for message in mgr.get_history():
                role = message.get('role')
                content = message.get('content', '')
                prefix = 'User' if role == 'user' else 'Assistant'
                chat_history.insert(tk.END, f"{prefix}: {content}\n\n")
        chat_history.see(tk.END)
        chat_history.configure(state='disabled')
    except Exception:
        pass


def refresh_list() -> None:
    models = mgr.list_models()
    try:
        listbox.delete(0, tk.END)
        for model_name in models:
            listbox.insert(tk.END, model_name)
    except Exception:
        pass


def pick_model() -> None:
    choice = listbox.get(tk.ACTIVE)
    if not choice:
        update_status('Select a model to load.')
        return
    loaded = mgr.load_model(choice)
    if loaded:
        render_history()
        update_status(f"Model loaded: {choice} ({getattr(mgr, 'backend', 'unknown')})")
    else:
        update_status(f"Failed to load: {choice}")


def add_model() -> None:
    file_path = filedialog.askopenfilename(title='Pick a model file', filetypes=[('Model files', '*.gguf')])
    if not file_path:
        return
    destination = Path('models') / Path(file_path).name
    try:
        shutil.copy(file_path, destination)
    except Exception as exc:
        update_status(f'Failed to add model: {exc}')
        return
    refresh_list()
    update_status(f'Added model: {destination.name}')


def append_user_message(message: str) -> None:
    try:
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f'User: {message}\n\n')
        chat_history.see(tk.END)
        chat_history.configure(state='disabled')
    except Exception:
        pass


def run_prompt() -> None:
    text = entry.get('1.0', tk.END).strip()
    if not text:
        return
    if not mgr.is_loaded():
        update_status('Load a model before sending a prompt.')
        return
    entry.delete('1.0', tk.END)
    append_user_message(text)
    update_status('Generating...')

    def worker(prompt: str) -> None:
        try:
            mgr.generate(prompt)
            error_message = None
        except Exception as exc:
            error_message = str(exc)

        def on_complete() -> None:
            if error_message:
                update_status(error_message)
            else:
                update_status('Done')
            render_history()

        try:
            root.after(0, on_complete)
        except Exception:
            on_complete()

    threading.Thread(target=worker, args=(text,), daemon=True).start()


def clear_chat() -> None:
    if not mgr.is_loaded():
        update_status('Load a model to clear its history.')
        return
    mgr.clear_history()
    render_history()
    update_status('Chat history cleared.')


def on_close() -> None:
    try:
        mgr.unload()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def build_ui() -> None:
    ctk.set_appearance_mode('dark')
    ctk.set_default_color_theme('blue')

    global root, listbox, status, chat_history, entry
    root = ctk.CTk()
    root.title('Offline LLM Switcher')
    root.geometry('900x700')
    root.configure(fg_color=BG_GRAD_TOP)

    # Left panel
    side_frame = ctk.CTkFrame(root, width=250, fg_color=BG_PANEL, corner_radius=12)
    side_frame.pack(side='left', fill='y', padx=10, pady=10)
    side_frame.pack_propagate(False)

    ctk.CTkLabel(side_frame, text='Models:', text_color=TEXT, font=FONT_H1).pack(pady=(24, 8))

    listbox_frame = ctk.CTkFrame(side_frame, fg_color=BG_PANEL, corner_radius=0)
    listbox_frame.pack(fill='both', expand=True)

    listbox = tk.Listbox(listbox_frame, bg=BG_LIST, fg=TEXT, selectbackground=HL_LIST, selectforeground=TEXT, relief=tk.FLAT, highlightthickness=0, font=FONT_UI)
    listbox.pack(side='left', fill='both', expand=True, padx=(0, 4), pady=8)

    btn_pad = dict(fill='x', pady=6)
    ctk.CTkButton(side_frame, text='Refresh', command=refresh_list, corner_radius=14, fg_color=ACCENT, hover_color='#6a9dff', text_color='white', font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame, text='Load Model', command=pick_model, corner_radius=14, fg_color=ACCENT, hover_color='#6a9dff', text_color='white', font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame, text='Add Model', command=add_model, corner_radius=14, fg_color=ACCENT, hover_color='#6a9dff', text_color='white', font=FONT_BOLD).pack(**btn_pad)

    status = ctk.CTkLabel(side_frame, text='No model loaded', text_color=MUTED, font=FONT_UI, anchor='center', justify='center')
    status.pack(pady=6, fill='x')

    # Right panel
    chat_frame = ctk.CTkFrame(root, fg_color=BG_PANEL, corner_radius=12)
    chat_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    chat_border = ctk.CTkFrame(chat_frame, fg_color=BORDER_ACCENT, corner_radius=12)
    chat_border.pack(fill='both', expand=True, padx=10, pady=(10, 6))

    chat_history = ctk.CTkTextbox(chat_border, height=420, corner_radius=10, fg_color=BG_LIST, text_color=TEXT, font=FONT_UI, wrap='word')
    chat_history.pack(fill='both', expand=True, padx=4, pady=4)
    chat_history.configure(state='disabled')

    entry_border = ctk.CTkFrame(chat_frame, fg_color=BORDER_ACCENT, corner_radius=12)
    entry_border.pack(fill='x', padx=10, pady=(0, 6))

    entry = ctk.CTkTextbox(entry_border, height=80, corner_radius=10, fg_color=BG_LIST, text_color=TEXT, font=FONT_UI, wrap='word')
    entry.pack(fill='x', padx=4, pady=4)

    ctk.CTkButton(chat_frame, text='Send', command=run_prompt, height=36, corner_radius=14, fg_color=ACCENT, hover_color='#8db9ff', text_color='white', font=FONT_BOLD).pack(fill='x', padx=10, pady=(0, 10))
    ctk.CTkButton(chat_frame, text='Clear History', command=clear_chat, height=34, corner_radius=14, fg_color='#2b374f', text_color=TEXT, font=FONT_UI).pack(fill='x', padx=10, pady=(0, 8))

    def toggle_side():
        if side_frame.winfo_ismapped():
            side_frame.pack_forget()
        else:
            side_frame.pack(side='left', fill='y', padx=10, pady=10, before=chat_frame)

    toggle_btn = ctk.CTkButton(root, text='◀', width=30, height=30, corner_radius=8, fg_color=ACCENT, hover_color='#8db9ff', text_color='white', font=FONT_BOLD, command=toggle_side)
    toggle_btn.place(x=10, y=10)

    refresh_list()
    render_history()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    build_ui()
