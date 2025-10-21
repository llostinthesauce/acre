import tkinter as tk
from tkinter import filedialog
from tkinter import *
import shutil
import threading
from model_manager import model_manager
import customtkinter as ctk
from PIL import Image, ImageTk

mgr = model_manager()

#Theme and Fonts
BG_GRAD_TOP   = "#0a1022"
BG_PANEL      = "#0e1730"
BG_LIST       = "#0b1328"
HL_LIST       = "#1c2c56"
TEXT          = "#e9f1ff"
MUTED         = "#a9b8d6"
ACCENT        = "#6ea5ff"
BORDER_ACCENT = "#91bbff"
FONT_UI = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_H1 = ("Segoe UI", 16, "bold")
PLACEHOLDER = "Ask me anything..."

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
            status.configure(text=f"Model loaded: {choice}", wraplength=230, justify="left")
        else:
            status.configure(text=f"Failed to load: {choice}", wraplength=230, justify="left")

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
    if not text or (text == PLACEHOLDER):
        return
    entry.delete("1.0", tk.END)
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"User: {text}\n")
    chat_history.see(tk.END)
    chat_history.configure(state='disabled')
    update_logo_visibility() 
    status.configure(text="Generating...")

    def task():
        try:
            output_text = mgr.generate(text)
        except Exception as e:
            output_text = f"Error: {e}"

        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Assistant: {output_text}\n\n")
        chat_history.see(tk.END)
        chat_history.configure(state='disabled')
        update_logo_visibility()
        status.configure(text="Done")

    threading.Thread(target=task, daemon=True).start()

#UI                                  
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Offline LLM Switcher")
root.geometry("900x700")
root.configure(fg_color=BG_GRAD_TOP)

# Left panel with padding
side_frame = ctk.CTkFrame(root, width=250, fg_color=BG_PANEL, corner_radius=12)
side_frame.pack(side="left", fill="y", padx=10, pady=10)
side_frame.pack_propagate(False)

ctk.CTkLabel(side_frame, text="Models:", text_color=TEXT, font=FONT_H1).pack(pady=(36,10))

listbox_frame = ctk.CTkFrame(side_frame, fg_color=BG_PANEL, corner_radius=0)
listbox_frame.pack(fill="both", expand=True)


listbox = tk.Listbox(
    listbox_frame, 
    bg=BG_LIST,
    fg=TEXT,
    selectbackground=HL_LIST,
    selectforeground=TEXT,
    relief=tk.FLAT,
    highlightthickness=0,
    font=FONT_UI,
)
listbox.pack(side="left", fill="both", expand=True, padx=(0,4))

btn_pad = dict(fill="x", pady=6)
ctk.CTkButton(
    side_frame, 
    text="Refresh", 
    command=refresh_list,
    corner_radius=14,
    fg_color=ACCENT,
    hover_color="#6a9dff",
    text_color="white",
    font=FONT_BOLD
).pack(**btn_pad)

ctk.CTkButton(
    side_frame, 
    text="Load Model", 
    command=pick_model,
    corner_radius=14,
    fg_color=ACCENT,
    hover_color="#6a9dff",
    text_color="white",
    font=FONT_BOLD
).pack(**btn_pad)

ctk.CTkButton(
    side_frame, 
    text="Add Model", 
    command=add_model,
    corner_radius=14,
    fg_color=ACCENT,
    hover_color="#6a9dff",
    text_color="white",
    font=FONT_BOLD
).pack(**btn_pad)

status = ctk.CTkLabel(
    side_frame, 
    text="No model loaded",  
    text_color=MUTED,
    font=FONT_UI,
    anchor="center",
    justify="center"
)
status.pack(pady=6, fill="x")

# Right panel
chat_frame = ctk.CTkFrame(root, fg_color=BG_PANEL, corner_radius=12)
chat_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

chat_border = ctk.CTkFrame(chat_frame, fg_color=BORDER_ACCENT, corner_radius=12)
chat_border.pack(fill="both", expand=True, padx=10, pady=(10, 6))

chat_history = ctk.CTkTextbox(
    chat_border, 
    height=420,
    corner_radius=10,
    fg_color=BG_LIST,
    text_color=TEXT,
    font=FONT_UI,
    wrap="word",
)
chat_history.pack(fill="both", expand=True, padx=4, pady=4)
chat_history.configure(state='disabled')

entry_border = ctk.CTkFrame(chat_frame, fg_color=BORDER_ACCENT, corner_radius=12)
entry_border.pack(fill="x", padx=10, pady=(0, 6))

entry_container = ctk.CTkFrame(entry_border, fg_color=BG_LIST, corner_radius=10)
entry_container.pack(fill="x", padx=4, pady=4)

try:
    logo_image = Image.open("transparent-logo.png")
    logo_image = logo_image.resize((200, 200))
    logo_tk = ImageTk.PhotoImage(logo_image)

    logo_label = tk.Label(chat_border, image=logo_tk, bg=BG_LIST)
    logo_label.image = logo_tk
    logo_label.place(relx=0.5, rely=0.5, anchor="center")
    logo_label.lift()
except Exception as e:
    print(f"Error loading logo image: {e}")
 

send_btn = ctk.CTkButton(
    entry_container, 
    text="↑", 
    width=44,
    height=36,
    corner_radius=10,
    fg_color=ACCENT,
    hover_color="#8db9ff",
    text_color="white",
    font=FONT_BOLD,
    command=run_prompt
)
send_btn.pack(side="right", padx=(4, 8), pady=6)

entry_frame = ctk.CTkFrame(entry_container, fg_color=BG_LIST, corner_radius=10)
entry_frame.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=6)

entry = tk.Text(
    entry_frame,
    height=1,
    wrap="word",
    bg=BG_LIST,
    fg=MUTED,
    insertbackground=TEXT,
    font=FONT_UI,
    relief="flat",
    highlightthickness=0,
    bd=0
)
entry.pack(fill="both", expand=True, padx=6, pady=4)

entry.insert("1.0", PLACEHOLDER)

def entry_focus_in(event):
    if entry.get("1.0", "end-1c") == PLACEHOLDER:
        entry.delete("1.0", "end")
        entry.configure(fg=TEXT)

def entry_focus_out(event):
    if entry.get("1.0", "end-1c").strip() == "":
        entry.insert("1.0", PLACEHOLDER)
        entry.configure(fg=MUTED)

def update_logo_visibility():
    content = chat_history.get("1.0", "end-1c").strip()
    if content:
        try:
            logo_label.place_forget()
        except:
            pass
    else:
        try:
            logo_label.place(relx=0.5, rely=0.4, anchor="center")
            logo.label.lift()
        except:
            pass

chat_history.bind("<KeyRelease>", lambda e: update_logo_visibility())

entry.bind("<FocusIn>", entry_focus_in)
entry.bind("<FocusOut>", entry_focus_out)

def auto_resize(event=None):
    lines = int(entry.index('end-1c').split('.')[0])
    entry.configure(height=min(max(1, lines), 8))

entry.bind("<KeyRelease>", auto_resize)


toggle_state = {"open": True}

toggle_btn = ctk.CTkButton(
    root,
    text="◀",
    width=30,
    height=30,
    corner_radius=8,
    fg_color=ACCENT,
    hover_color="#8db9ff",
    text_color="white",
    font=FONT_BOLD,
    command=lambda: (
        (
            side_frame.pack_forget(),
            toggle_btn.configure(text="▶"),
            toggle_state.update(open=False)
        )
        if toggle_state["open"]
        else (
            side_frame.pack(side="left", fill="y", padx=10, pady=10, before=chat_frame),
            toggle_btn.configure(text="◀"),
            toggle_state.update(open=True)
        ),
        toggle_btn.lift()
    ),
)
toggle_btn.place(x=10, y=10)
toggle_btn.lift()

refresh_list()
root.mainloop()