import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext

from model_manager import ModelManager


mgr = ModelManager()


def update_status(message: str) -> None:
    status.config(text=message, wraplength=230, justify="left")


def render_history() -> None:
    chat_history.configure(state="normal")
    chat_history.delete("1.0", tk.END)

    if mgr.is_loaded():
        for message in mgr.get_history():
            role = message.get("role")
            content = message.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            chat_history.insert(tk.END, f"{prefix}: {content}\n\n")

    chat_history.see(tk.END)
    chat_history.configure(state="disabled")


def refresh_list() -> None:
    models = mgr.list_models()
    listbox.delete(0, tk.END)
    for model_name in models:
        listbox.insert(tk.END, model_name)


def pick_model() -> None:
    choice = listbox.get(tk.ACTIVE)
    if not choice:
        update_status("Select a model to load.")
        return

    loaded = mgr.load_model(choice)
    if loaded:
        render_history()
        update_status(f"Model loaded: {choice} ({mgr.backend})")
    else:
        update_status(f"Failed to load: {choice}")


def add_model() -> None:
    file_path = filedialog.askopenfilename(
        title="Pick a model file",
        filetypes=[("Model files", "*.gguf")],
    )
    if not file_path:
        return

    destination = Path("models") / Path(file_path).name
    try:
        shutil.copy(file_path, destination)
    except Exception as exc:
        update_status(f"Failed to add model: {exc}")
        return

    refresh_list()
    update_status(f"Added model: {destination.name}")


def append_user_message(message: str) -> None:
    chat_history.configure(state="normal")
    chat_history.insert(tk.END, f"User: {message}\n\n")
    chat_history.see(tk.END)
    chat_history.configure(state="disabled")


def run_prompt() -> None:
    text = entry.get("1.0", tk.END).strip()
    if not text:
        return
    if not mgr.is_loaded():
        update_status("Load a model before sending a prompt.")
        return

    entry.delete("1.0", tk.END)
    append_user_message(text)
    update_status("Generating...")

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
                update_status("Done")
            render_history()

        root.after(0, on_complete)

    threading.Thread(target=worker, args=(text,), daemon=True).start()


def clear_chat() -> None:
    if not mgr.is_loaded():
        update_status("Load a model to clear its history.")
        return

    mgr.clear_history()
    render_history()
    update_status("Chat history cleared.")


def on_close() -> None:
    try:
        mgr.unload()
    finally:
        root.destroy()


root = tk.Tk()
root.title("Offline LLM Switcher")
root.geometry("900x700")

side_frame = tk.Frame(root, width=250, padx=10, pady=10)
side_frame.pack(side=tk.LEFT, fill=tk.Y)
side_frame.pack_propagate(False)

tk.Label(side_frame, text="Models:").pack(pady=(0, 5))

listbox_frame = tk.Frame(side_frame)
listbox_frame.pack(fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
listbox.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=listbox.yview)

tk.Button(side_frame, text="Refresh", command=refresh_list).pack(fill=tk.X, pady=2)
tk.Button(side_frame, text="Load Model", command=pick_model).pack(fill=tk.X, pady=2)
tk.Button(side_frame, text="Add Model", command=add_model).pack(fill=tk.X, pady=2)

status = tk.Label(side_frame, text="No model loaded", anchor="w", justify="left", wraplength=230)
status.pack(pady=10)

chat_frame = tk.Frame(root, padx=10, pady=10)
chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

chat_history = scrolledtext.ScrolledText(chat_frame, state="disabled", wrap=tk.WORD)
chat_history.pack(fill=tk.BOTH, expand=True)

entry = tk.Text(chat_frame, height=4)
entry.pack(fill=tk.X, pady=5)

tk.Button(chat_frame, text="Send", command=run_prompt).pack(fill=tk.X, pady=5)
tk.Button(chat_frame, text="Clear History", command=clear_chat).pack(fill=tk.X, pady=5)

refresh_list()
render_history()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
