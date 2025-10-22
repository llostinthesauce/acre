import json
import base64
import secrets
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Optional
import customtkinter as ctk
from PIL import Image, ImageTk
from model_manager import ModelManager

BG_GRAD_TOP="#0a1022"
BG_PANEL="#0e1730"
BG_LIST="#0b1328"
HL_LIST="#1c2c56"
TEXT="#e9f1ff"
MUTED="#a9b8d6"
ACCENT="#6ea5ff"
BORDER_ACCENT="#91bbff"
FONT_UI=("Segoe UI",11)
FONT_BOLD=("Segoe UI",11,"bold")
FONT_H1=("Segoe UI",18,"bold")
FONT_H2=("Segoe UI",16,"bold")
PLACEHOLDER="Ask me anything..."
DISCLAIMER_TEXT=(
"WARNING — Read Carefully:\n\n"
"This application runs offline language models that do NOT have access to real-time information. "
"Models were trained on data available up to their release date and may provide inaccurate, "
"outdated, or incomplete answers. They can also reflect biases present in their training data. "
"Under NO circumstances should responses from these models be used for life-critical, medical, "
"legal, or safety-related decisions. Always verify important information with qualified professionals "
"or authoritative sources. The developers provide this tool for research and experimentation only; "
"we do not assume responsibility for decisions made based on model output.\n\n"
"By checking the box below and clicking 'I Understand', you acknowledge that you have read this "
"disclaimer, understand the limitations, and accept responsibility for how you use the system."
).replace("\\n", "\n")
CONFIG_PATH=Path("config/settings.json")

def _b64e(b:bytes)->str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")
def _b64d(s:str)->bytes:
    pad="="*(-len(s)%4)
    return base64.urlsafe_b64decode((s+pad).encode("utf-8"))
def load_settings()->dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}
def save_settings(data:dict)->None:
    CONFIG_PATH.parent.mkdir(parents=True,exist_ok=True)
    tmp=CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(CONFIG_PATH)
def have_password(settings:dict)->bool:
    return bool(settings.get("password_hash") and settings.get("salt") and settings.get("iterations"))
def required_username(settings:dict)->Optional[str]:
    u=settings.get("username")
    if isinstance(u,str) and u.strip():
        return u.strip()
    return None
def set_credentials(username:str,password:str,iterations:int=200000)->dict:
    if not username.strip():
        raise ValueError("Username cannot be empty.")
    if len(password)<6:
        raise ValueError("Password must be at least 6 characters.")
    salt=secrets.token_bytes(16)
    dk=pbkdf2_hmac("sha256",password.encode("utf-8"),salt,iterations)
    return {"username":username.strip(),"password_hash":_b64e(dk),"salt":_b64e(salt),"iterations":iterations}
def verify_password(pw:str,settings:dict)->bool:
    if have_password(settings):
        try:
            h=_b64d(settings["password_hash"])
            s=_b64d(settings["salt"])
            iters=int(settings["iterations"])
            k=pbkdf2_hmac("sha256",pw.encode("utf-8"),s,iters)
            return k==h
        except Exception:
            return False
    return False

mgr=None
root=None
side_frame=None
chat_frame=None
chat_history=None
listbox=None
entry=None
status=None
logo_label=None
toggle_btn=None
gate_frame=None
login_frame=None
disc_frame=None
setup_frame=None

def update_status(message:str)->None:
    if status:
        status.configure(text=message,wraplength=230,justify="left")
def render_history()->None:
    if not chat_history:
        return
    chat_history.configure(state="normal")
    chat_history.delete("1.0",tk.END)
    if mgr and mgr.is_loaded():
        for m in mgr.get_history():
            role=m.get("role")
            content=m.get("content","")
            prefix="User" if role=="user" else "Assistant"
            chat_history.insert(tk.END,f"{prefix}: {content}\n\n")
    chat_history.see(tk.END)
    chat_history.configure(state="disabled")
    update_logo_visibility()
def refresh_list()->None:
    if not (mgr and listbox):
        return
    models=mgr.list_models()
    listbox.delete(0,tk.END)
    for name in models:
        listbox.insert(tk.END,name)
def pick_model()->None:
    if not (mgr and listbox):
        return
    choice=listbox.get(tk.ACTIVE)
    if not choice:
        update_status("Select a model to load.")
        return
    loaded=mgr.load_model(choice)
    if loaded:
        render_history()
        backend=getattr(mgr,"backend",None)
        update_status(f"Model loaded: {choice} ({backend})" if backend else f"Model loaded: {choice}")
    else:
        update_status(f"Failed to load: {choice}")
def add_model() -> None:
    import os
    choice_is_folder = messagebox.askyesno(
        "Import Model",
        "Import a model FOLDER? (Yes = Folder, No = Single File)"
    )

    Path("models").mkdir(parents=True, exist_ok=True)

    if choice_is_folder:
        dirpath = filedialog.askdirectory(title="Pick a model folder")
        if not dirpath:
            return
        name = os.path.basename(dirpath.rstrip("/"))
        dest = Path("models") / name
        if dest.exists():
            messagebox.showerror("Already Exists", f"{dest} already exists.")
            return
        try:
            shutil.copytree(dirpath, dest)
        except Exception as exc:
            update_status(f"Failed to add folder: {exc}")
            return
        refresh_list()
        update_status(f"Added model folder: {dest.name}")
        return

    file_path = filedialog.askopenfilename(
        title="Pick a model file",
        filetypes=[("Model files", "*.gguf"), ("All files", "*.*")]
    )
    if not file_path:
        return
    dest = Path("models") / Path(file_path).name
    try:
        shutil.copy(file_path, dest)
    except Exception as exc:
        update_status(f"Failed to add file: {exc}")
        return
    refresh_list()
    update_status(f"Added model file: {dest.name}")
def append_user_message(message:str)->None:
    if not chat_history:
        return
    chat_history.configure(state="normal")
    chat_history.insert(tk.END,f"User: {message}\n\n")
    chat_history.see(tk.END)
    chat_history.configure(state="disabled")
    update_logo_visibility()
def run_prompt()->None:
    if not entry:
        return
    text=entry.get("1.0",tk.END).strip()
    if not text or text==PLACEHOLDER:
        return
    if not (mgr and mgr.is_loaded()):
        update_status("Load a model before sending a prompt.")
        return
    entry.delete("1.0",tk.END)
    append_user_message(text)
    update_status("Generating...")
    def worker(prompt:str)->None:
        err=None
        try:
            mgr.generate(prompt)
        except Exception as exc:
            err=str(exc)
        def done()->None:
            update_status(err if err else "Done")
            render_history()
        root.after(0,done)
    threading.Thread(target=worker,args=(text,),daemon=True).start()
def clear_chat()->None:
    if not (mgr and mgr.is_loaded()):
        update_status("Load a model to clear its history.")
        return
    mgr.clear_history()
    render_history()
    update_status("Chat history cleared.")
def logout_action()->None:
    try:
        if mgr:
            mgr.unload()
    except Exception:
        pass
    teardown_main_ui()
    build_gate_ui()
def update_logo_visibility()->None:
    global logo_label
    if logo_label is None or chat_history is None:
        return
    content=chat_history.get("1.0","end-1c").strip()
    if content:
        try:
            logo_label.place_forget()
        except Exception:
            pass
    else:
        try:
            logo_label.place(relx=0.5,rely=0.4,anchor="center")
            logo_label.lift()
        except Exception:
            pass
def build_main_ui()->None:
    global side_frame,chat_frame,chat_history,listbox,entry,status,toggle_btn,logo_label,mgr
    mgr=ModelManager()
    side_frame=ctk.CTkFrame(root,width=250,fg_color=BG_PANEL,corner_radius=12)
    side_frame.pack(side="left",fill="y",padx=10,pady=10)
    side_frame.pack_propagate(False)
    ctk.CTkLabel(side_frame,text="Models:",text_color=TEXT,font=FONT_H2).pack(pady=(36,10))
    listbox_frame=ctk.CTkFrame(side_frame,fg_color=BG_PANEL,corner_radius=0)
    listbox_frame.pack(fill="both",expand=True)
    global listbox
    listbox=tk.Listbox(listbox_frame,bg=BG_LIST,fg=TEXT,selectbackground=HL_LIST,selectforeground=TEXT,relief=tk.FLAT,highlightthickness=0,font=FONT_UI)
    listbox.pack(side="left",fill="both",expand=True,padx=(0,4))
    btn_pad=dict(fill="x",pady=6)
    ctk.CTkButton(side_frame,text="Refresh",command=refresh_list,corner_radius=14,fg_color=ACCENT,hover_color="#6a9dff",text_color="white",font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame,text="Load Model",command=pick_model,corner_radius=14,fg_color=ACCENT,hover_color="#6a9dff",text_color="white",font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame,text="Add Model",command=add_model,corner_radius=14,fg_color=ACCENT,hover_color="#6a9dff",text_color="white",font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame,text="Clear History",command=clear_chat,corner_radius=14,fg_color="#415a8f",hover_color="#5674b3",text_color="white",font=FONT_BOLD).pack(**btn_pad)
    ctk.CTkButton(side_frame,text="Logout",command=logout_action,corner_radius=14,fg_color="#b84d4d",hover_color="#d85f5f",text_color="white",font=FONT_BOLD).pack(**btn_pad)
    s=ctk.CTkLabel(side_frame,text="No model loaded",text_color=MUTED,font=FONT_UI,anchor="center",justify="center")
    s.pack(pady=6,fill="x")
    global status
    status=s
    chat_frame_local=ctk.CTkFrame(root,fg_color=BG_PANEL,corner_radius=12)
    chat_frame_local.pack(side="right",fill="both",expand=True,padx=10,pady=10)
    global chat_frame
    chat_frame=chat_frame_local
    chat_border=ctk.CTkFrame(chat_frame_local,fg_color=BORDER_ACCENT,corner_radius=12)
    chat_border.pack(fill="both",expand=True,padx=10,pady=(10,6))
    global chat_history
    chat_history=ctk.CTkTextbox(chat_border,height=420,corner_radius=10,fg_color=BG_LIST,text_color=TEXT,font=FONT_UI,wrap="word")
    chat_history.pack(fill="both",expand=True,padx=4,pady=4)
    chat_history.configure(state='disabled')
    global logo_label
    logo_label=None
    try:
        p=Path("transparent-logo.png")
        if p.exists():
            img=Image.open(p).resize((200,200))
            logo_tk=ImageTk.PhotoImage(img)
            logo_label=tk.Label(chat_border,image=logo_tk,bg=BG_LIST)
            logo_label.image=logo_tk
            logo_label.place(relx=0.5,rely=0.5,anchor="center")
            logo_label.lift()
    except Exception:
        logo_label=None
    entry_border=ctk.CTkFrame(chat_frame_local,fg_color=BORDER_ACCENT,corner_radius=12)
    entry_border.pack(fill="x",padx=10,pady=(0,6))
    entry_container=ctk.CTkFrame(entry_border,fg_color=BG_LIST,corner_radius=10)
    entry_container.pack(fill="x",padx=4,pady=4)
    send_btn=ctk.CTkButton(entry_container,text="↑",width=44,height=36,corner_radius=10,fg_color=ACCENT,hover_color="#8db9ff",text_color="white",font=FONT_BOLD,command=run_prompt)
    send_btn.pack(side="right",padx=(4,8),pady=6)
    entry_frame=ctk.CTkFrame(entry_container,fg_color=BG_LIST,corner_radius=10)
    entry_frame.pack(side="left",fill="both",expand=True,padx=(8,4),pady=6)
    global entry
    entry=tk.Text(entry_frame,height=1,wrap="word",bg=BG_LIST,fg=MUTED,insertbackground=TEXT,font=FONT_UI,relief="flat",highlightthickness=0,bd=0)
    entry.pack(fill="both",expand=True,padx=6,pady=4)
    entry.insert("1.0",PLACEHOLDER)
    def _in(_):
        if entry.get("1.0","end-1c")==PLACEHOLDER:
            entry.delete("1.0","end")
            entry.configure(fg=TEXT)
    def _out(_):
        if entry.get("1.0","end-1c").strip()=="":
            entry.insert("1.0",PLACEHOLDER)
            entry.configure(fg=MUTED)
    entry.bind("<FocusIn>",_in)
    entry.bind("<FocusOut>",_out)
    def auto(_=None):
        lines=int(entry.index('end-1c').split('.')[0])
        entry.configure(height=min(max(1,lines),8))
    entry.bind("<KeyRelease>",auto)
    tstate={"open":True}
    def _toggle():
        if tstate["open"]:
            side_frame.pack_forget()
            toggle_btn.configure(text="▶")
            tstate["open"]=False
        else:
            side_frame.pack(side="left",fill="y",padx=10,pady=10,before=chat_frame_local)
            toggle_btn.configure(text="◀")
            tstate["open"]=True
        toggle_btn.lift()
    global toggle_btn
    toggle_btn=ctk.CTkButton(root,text="◀",width=30,height=30,corner_radius=8,fg_color=ACCENT,hover_color="#8db9ff",text_color="white",font=FONT_BOLD,command=_toggle)
    toggle_btn.place(x=10,y=10)
    toggle_btn.lift()
    refresh_list()
    render_history()
    update_status("No model loaded")
def teardown_main_ui()->None:
    global side_frame,chat_frame,chat_history,listbox,entry,status,toggle_btn,logo_label
    for w in (side_frame,chat_frame,toggle_btn):
        try:
            if w is not None:
                w.destroy()
        except Exception:
            pass
    side_frame=chat_frame=chat_history=listbox=entry=status=toggle_btn=logo_label=None
def build_gate_ui()->None:
    settings=load_settings()
    first_run=not have_password(settings) or required_username(settings) is None
    global gate_frame,setup_frame,login_frame,disc_frame
    gate_frame=ctk.CTkFrame(root,fg_color=BG_PANEL,corner_radius=14)
    gate_frame.place(relx=0.5,rely=0.5,anchor="center",relwidth=0.92,relheight=0.92)
    setup_frame=ctk.CTkFrame(gate_frame,fg_color=BG_PANEL,corner_radius=14)
    login_frame=ctk.CTkFrame(gate_frame,fg_color=BG_PANEL,corner_radius=14)
    disc_frame=ctk.CTkFrame(gate_frame,fg_color=BG_PANEL,corner_radius=14)
    for f in (setup_frame,login_frame,disc_frame):
        f.place(relx=0.5,rely=0.5,anchor="center",relwidth=0.96,relheight=0.96)
    def show(f):
        f.lift()
    ctk.CTkLabel(setup_frame,text="Initial Setup",font=FONT_H1,text_color=TEXT).pack(pady=(16,8))
    ctk.CTkLabel(setup_frame,text="Create your username and password.",font=FONT_UI,text_color=MUTED).pack(pady=(0,8))
    setup_form=ctk.CTkFrame(setup_frame,fg_color=BG_PANEL)
    setup_form.pack(pady=8)
    new_user_var=tk.StringVar(value=required_username(settings) or "")
    new_pass_var=tk.StringVar(value="")
    new_pass2_var=tk.StringVar(value="")
    rowpad=dict(padx=6,pady=6,sticky="w")
    ctk.CTkLabel(setup_form,text="Username",text_color=TEXT,font=FONT_UI).grid(row=0,column=0,**rowpad)
    ctk.CTkEntry(setup_form,textvariable=new_user_var,width=260).grid(row=0,column=1,**rowpad)
    ctk.CTkLabel(setup_form,text="Password",text_color=TEXT,font=FONT_UI).grid(row=1,column=0,**rowpad)
    ctk.CTkEntry(setup_form,textvariable=new_pass_var,show="*",width=260).grid(row=1,column=1,**rowpad)
    ctk.CTkLabel(setup_form,text="Confirm Password",text_color=TEXT,font=FONT_UI).grid(row=2,column=0,**rowpad)
    ctk.CTkEntry(setup_form,textvariable=new_pass2_var,show="*",width=260).grid(row=2,column=1,**rowpad)
    def do_setup():
        u=new_user_var.get().strip()
        p1=new_pass_var.get()
        p2=new_pass2_var.get()
        if p1!=p2:
            messagebox.showerror("Setup Failed","Passwords do not match.")
            return
        try:
            creds=set_credentials(u,p1)
        except Exception as e:
            messagebox.showerror("Setup Failed",str(e))
            return
        s=load_settings()
        s.update(creds)
        try:
            save_settings(s)
        except Exception as e:
            messagebox.showerror("Setup Failed",f"Could not save settings: {e}")
            return
        disc_frame.lift()
    ctk.CTkButton(setup_frame,text="Save & Continue",corner_radius=12,fg_color=ACCENT,text_color="white",font=FONT_BOLD,command=do_setup).pack(pady=12)
    ctk.CTkLabel(login_frame,text="ACRE LLM Switchboard",font=FONT_H1,text_color=TEXT).pack(pady=(24,8))
    ctk.CTkLabel(login_frame,text="Please log in to continue",font=FONT_UI,text_color=MUTED).pack(pady=(0,16))
    form=ctk.CTkFrame(login_frame,fg_color=BG_PANEL)
    form.pack(pady=4)
    expected_user=required_username(settings) or "admin"
    user_var=tk.StringVar(value=expected_user)
    pass_var=tk.StringVar(value="")
    ctk.CTkLabel(form,text="Username",text_color=TEXT,font=FONT_UI).grid(row=0,column=0,**rowpad)
    ctk.CTkEntry(form,textvariable=user_var,width=260).grid(row=0,column=1,**rowpad)
    ctk.CTkLabel(form,text="Password",text_color=TEXT,font=FONT_UI).grid(row=1,column=0,**rowpad)
    pass_entry=ctk.CTkEntry(form,textvariable=pass_var,show="*",width=260)
    pass_entry.grid(row=1,column=1,**rowpad)
    def do_login():
        u=user_var.get().strip()
        p=pass_var.get().strip()
        configured=required_username(load_settings())
        if configured and u!=configured:
            messagebox.showerror("Login Failed","Invalid username.")
            pass_var.set("")
            pass_entry.focus_set()
            return
        if verify_password(p,load_settings()):
            pass_var.set("")
            disc_frame.lift()
        else:
            messagebox.showerror("Login Failed","Invalid credentials. Please try again.")
            pass_var.set("")
            pass_entry.focus_set()
    ctk.CTkButton(login_frame,text="Login",corner_radius=12,fg_color=ACCENT,text_color="white",font=FONT_BOLD,command=do_login).pack(pady=18)
    ctk.CTkLabel(disc_frame,text="IMPORTANT DISCLAIMER",font=FONT_H1,text_color="#ffcccc").pack(pady=(16,6))
    wrap=ctk.CTkFrame(disc_frame,fg_color=BG_PANEL)
    wrap.pack(fill="both",expand=True,padx=12,pady=(0,8))
    scrollbar=ctk.CTkScrollbar(wrap)
    scrollbar.pack(side="right",fill="y",padx=(0,4))
    disc_tb=ctk.CTkTextbox(wrap,corner_radius=10,fg_color=BG_LIST,text_color=TEXT,font=FONT_UI,wrap="word")
    disc_tb.pack(side="left",fill="both",expand=True,padx=(4,0),pady=4)
    disc_tb.insert("1.0",DISCLAIMER_TEXT)
    disc_tb.configure(state="disabled")
    disc_tb.configure(yscrollcommand=scrollbar.set)
    scrollbar.configure(command=disc_tb.yview)
    btn_row=ctk.CTkFrame(disc_frame,fg_color=BG_PANEL)
    btn_row.pack(pady=6)
    agree_var=tk.BooleanVar(value=False)
    def on_check():
        understand_btn.configure(state=("normal" if agree_var.get() else "disabled"))
    ctk.CTkCheckBox(btn_row,text="I have read and understand the disclaimer",variable=agree_var,command=on_check,text_color=TEXT,fg_color=ACCENT,hover_color="#8db9ff").grid(row=0,column=0,padx=6,pady=6)
    def accept():
        try:
            gate_frame.destroy()
        except Exception:
            pass
        Path("models").mkdir(parents=True,exist_ok=True)
        Path("history").mkdir(parents=True,exist_ok=True)
        build_main_ui()
    understand_btn=ctk.CTkButton(btn_row,text="I Understand",state="disabled",corner_radius=12,fg_color=ACCENT,text_color="white",font=FONT_BOLD,command=accept)
    understand_btn.grid(row=0,column=1,padx=6,pady=6)
    back_target=login_frame if not first_run else setup_frame
    ctk.CTkButton(btn_row,text="Back",corner_radius=12,fg_color="#415a8f",text_color="white",font=FONT_BOLD,command=lambda: back_target.lift()).grid(row=0,column=2,padx=6,pady=6)
    (setup_frame if first_run else login_frame).lift()
def on_close()->None:
    try:
        if mgr:
            mgr.unload()
    finally:
        root.destroy()
def main()->None:
    global root
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root=ctk.CTk()
    root.title("Offline LLM Switcher")
    root.geometry("900x700")
    root.configure(fg_color=BG_GRAD_TOP)
    build_gate_ui()
    root.protocol("WM_DELETE_WINDOW",on_close)
    root.mainloop()
if __name__=="__main__":
    main()
