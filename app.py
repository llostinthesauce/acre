import tkinter as tk
from tkinter import filedialog, scrolledtext
import shutil
import threading
from model_manager import model_manager

mgr = model_manager()

def refresh_list():
    models = mgr.list_models()
    listbox.delete(0, tk.END)
    for m in models:
        listbox.insert(tk.END, m)

def pick_model():
    choice = listbox.get(tk.ACTIVE)
    if choice:
        loaded = mgr.load_model(choice)
        if loaded:
            status.config(text=f"Model loaded: {choice}", wraplength=230, justify="left")
        else:
            status.config(text=f"Failed to load: {choice}", wraplength=230, justify="left")

def add_model():
    file_path = filedialog.askopenfilename(
        title="Pick a model file",
        filetypes=[("Model files", "*.gguf *.bin *.safetensors")]
    )
    if file_path:
        shutil.copy(file_path, "models")
        refresh_list()

def run_prompt():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        return
    entry.delete("1.0", tk.END)
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"User: {text}\n")
    chat_history.see(tk.END)
    chat_history.configure(state='disabled')
    status.config(text="Generating...")

    def task():
        try:
            output_text = mgr.generate(text)
        except Exception as e:
            output_text = f"Error: {e}"

        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Assistant: {output_text}\n\n")
        chat_history.see(tk.END)
        chat_history.configure(state='disabled')
        status.config(text="Done")

    threading.Thread(target=task, daemon=True).start()

root = tk.Tk()
root.title("Offline LLM Switcher")
root.geometry("900x700")

# Left panel with padding
side_frame = tk.Frame(root, width=250, padx=10, pady=10)
side_frame.pack(side=tk.LEFT, fill=tk.Y)
side_frame.pack_propagate(False)

tk.Label(side_frame, text="Models:").pack(pady=(0,5))
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

# Right panel
chat_frame = tk.Frame(root, padx=10, pady=10)
chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

chat_history = scrolledtext.ScrolledText(chat_frame, state='disabled', wrap=tk.WORD)
chat_history.pack(fill=tk.BOTH, expand=True)

entry = tk.Text(chat_frame, height=4)
entry.pack(fill=tk.X, pady=5)

tk.Button(chat_frame, text="Send", command=run_prompt).pack(fill=tk.X, pady=5)

refresh_list()
root.mainloop()
