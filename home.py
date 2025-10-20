import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import sys
import os

"""
notesss: all dimensions are totally subjective and arbitrary rn, just gave em values to give em!
"""

#Color scheme basics --> sorta kinda along same lines as what Emma was working on!
BG_COLOR = "#0b2545"       
TEXT_COLOR = "#cfe8ff"      
BUTTON_BG = "#174a7f"       
BUTTON_FG = "#ffffff"

# Functions
# ---------------------------
def show_frame(frame):
    frame.tkraise()

# Login Handling --> LOCAL stuff for rn, will flesh out security later
# -----------------------
def check_login():
    user = username_var.get().strip()
    pw = password_var.get().strip()

    if user == "admin" and pw == "root":
        password_var.set("")  
        show_frame(disclaimer_frame)
    else:
        messagebox.showerror("Login Failed", "Invalid credentials. Please try again.")
        password_var.set("")
        password_entry.focus_set()



# Pop-Up disclaimer for users
# -----------------------
def disclaimer_checked_changed():
    if disclaimer_checkbox_var.get():
        understand_btn.config(state="normal")
    else:
        understand_btn.config(state="disabled")

def accept_disclaimer():
    show_frame(home_frame)



# actual connection to app.py --> somewhat buggy, am working on it --Rayyan
# -----------------------
def start_application():
    # directory checks!
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    if not os.path.exists(app_path):
        messagebox.showerror("Start Failed", "Could not find app.py in the project folder.")
        return
    python_exec = sys.executable

    try:
        # try start!
        subprocess.Popen([python_exec, app_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        messagebox.showerror("Start Failed", f"Failed to launch app.py:\n{e}")








# Main Window Setup
# ---------------------------
root = tk.Tk()
root.title("ACRE - LLM Switchboard")
root.geometry("520x420")
root.configure(bg=BG_COLOR)
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

login_frame = tk.Frame(root, bg=BG_COLOR)
disclaimer_frame = tk.Frame(root, bg=BG_COLOR)
home_frame = tk.Frame(root, bg=BG_COLOR)

for f in (login_frame, disclaimer_frame, home_frame):
    f.grid(row=0, column=0, sticky="nsew")


# Login frame --> variable stuff here, created it to match somewhat established UI in chat window, very change-able!
# -----------------------
username_var = tk.StringVar()
password_var = tk.StringVar()


title_lbl = tk.Label(login_frame, text="ACRE LLM Switchboard", font=("Arial", 20, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
title_lbl.pack(pady=(20,6))

sub_lbl = tk.Label(login_frame, text="Please log in to continue", font=("Arial", 12), bg=BG_COLOR, fg=TEXT_COLOR)
sub_lbl.pack(pady=(0,12))


form_container = tk.Frame(login_frame, bg=BG_COLOR)
form_container.pack(pady=10)

tk.Label(form_container, text="Username:", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, sticky="w", padx=5, pady=4)
username_entry = tk.Entry(form_container, textvariable=username_var, width=30)
username_entry.grid(row=0, column=1, padx=5, pady=4)
username_entry.focus_set()

tk.Label(form_container, text="Password:", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, sticky="w", padx=5, pady=4)
password_entry = tk.Entry(form_container, textvariable=password_var, width=30, show="*")

password_entry.grid(row=1, column=1, padx=5, pady=4)

login_btn = tk.Button(login_frame, text="Login", width=18, bg=BUTTON_BG, fg=BUTTON_FG, command=check_login)
login_btn.pack(pady=14)


# UPDATED BIG BIG DISCLAIMERRRR [spookyyyy] [made it scroll-able for professionalism]
# ---------------------------
disc_title = tk.Label(disclaimer_frame, text="IMPORTANT DISCLAIMER", font=("Arial", 16, "bold"), bg=BG_COLOR, fg="#ffcccc")
disc_title.pack(pady=(12,6))


disc_text = scrolledtext.ScrolledText(disclaimer_frame, wrap="word", height=12, width=60, bg="#07203a", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
disc_text.pack(padx=12, pady=(0,8))

long_disclaimer = (
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
)

disc_text.insert("1.0", long_disclaimer)
disc_text.config(state="disabled")  

checkbox_frame = tk.Frame(disclaimer_frame, bg=BG_COLOR)
checkbox_frame.pack(pady=(6,12))


disclaimer_checkbox_var = tk.BooleanVar(value=False)
disclaimer_checkbox = tk.Checkbutton(checkbox_frame, text="I have read the disclaimer", variable=disclaimer_checkbox_var, onvalue=True, offvalue=False, command=disclaimer_checked_changed, bg=BG_COLOR, fg=TEXT_COLOR, activebackground=BG_COLOR, selectcolor=BG_COLOR)

disclaimer_checkbox.grid(row=0, column=0, padx=8)

understand_btn = tk.Button(checkbox_frame, text="I Understand", width=14, bg=BUTTON_BG, fg=BUTTON_FG, state="disabled", command=accept_disclaimer)

understand_btn.grid(row=0, column=1, padx=8)


back_login_btn = tk.Button(checkbox_frame, text="Back to Login", width=12, command=lambda: show_frame(login_frame))

back_login_btn.grid(row=0, column=2, padx=8)


# UPDATED Home Screen Layout --> same thing about trying to match UI stuff 
# ---------------------------
home_header = tk.Label(home_frame, text="ACRE - Home", font=("Arial", 18, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
home_header.pack(pady=(18,6))

home_sub = tk.Label(home_frame, text="Ready to launch the offline LLM application.", font=("Arial", 12), bg=BG_COLOR, fg=TEXT_COLOR)
home_sub.pack(pady=(0,10))

start_app_btn = tk.Button(home_frame, text="Start Application", width=20, height=2, bg="#3b82c4", fg=BUTTON_FG, font=("Arial", 12, "bold"), command=start_application)
start_app_btn.pack(pady=12)

logout_btn = tk.Button(home_frame, text="Logout", width=12, command=lambda: show_frame(login_frame))
logout_btn.pack(pady=6)



#UPDATED Footer (development version) --> looks profesh i dunno 
# ---------------------------
ver_lbl = tk.Label(home_frame, text="Version 0.1 - Development Build", font=("Arial", 9), bg=BG_COLOR, fg="#9fcffb")
ver_lbl.pack(side="bottom", pady=10)



# STARTs
# ---------------------------
show_frame(login_frame)
root.mainloop()