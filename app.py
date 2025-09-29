import tkinter as tk
from tkinter import filedialog
import shutil
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
        mgr.load_model(choice)
        status.config(text=f"model loaded: {choice}")

def run_prompt():
    text = entry.get("1.0", tk.END).strip()
    if text:
        result = mgr.generate(text)
        output.delete("1.0", tk.END)
        output.insert(tk.END, result)

def add_model():
    file_path = filedialog.askopenfilename(
        title="pick a model file",
        filetypes=[("model files", "*.gguf *.bin")]
    )
    if file_path:
        shutil.copy(file_path, "models")
        refresh_list()

root = tk.Tk()
root.title("offline llm switcher")

listbox = tk.Listbox(root, width=40)
listbox.pack(pady=5)

refresh_btn = tk.Button(root, text="refresh", command=refresh_list)
refresh_btn.pack()

load_btn = tk.Button(root, text="load model", command=pick_model)
load_btn.pack()

entry = tk.Text(root, height=5, width=60)
entry.pack(pady=5)

run_btn = tk.Button(root, text="run prompt", command=run_prompt)
run_btn.pack()

output = tk.Text(root, height=10, width=60)
output.pack(pady=5)

status = tk.Label(root, text="no model loaded")
status.pack()

add_btn = tk.Button(root, text="add model", command=add_model)
add_btn.pack(pady=5)

refresh_list()
root.mainloop()