import tkinter as tk
from tkinter import messagebox

"""
notesss: all dimensions are totally subjective and arbitrary rn, just gave em values to give em!
"""

# Functions
# ---------------------------
def start_chat():
    """
    want to implement actual connection to app.py here at some point!
    """
    messagebox.showinfo("Start Chat, would open actual chata and model loading window here")

# Main Window Setup
# ---------------------------
root = tk.Tk()
root.title("ACRE - LLM Switchboard")
root.geometry("400x300") #subjective stuff here, can change to whateva
root.configure(bg="#f0f0f0")



# BIG BIG DISCLAIMERRRR [spookyyyy]
# ---------------------------
messagebox.showwarning(
    "IMPORTANT DISCLAIMER",
    "WARNING: The models used in this application are fully offline and may not provide accurate or up-to-date information. "
    "Users must verify any critical information independently. "
    "This software is provided as a research/demo tool and is not guaranteed for real-world decisions."
)

# Home Screen Layout [ROUGH] --> throwing stuff together to see waht looks good hehehehehe
# ---------------------------
header = tk.Label(root, text="ACRE LLM Switchboard", font=("Arial", 18, "bold"), fg="#333333",bg="#f0f0f0" )
header.pack(pady=(20, 5))

separator = tk.Frame(root, height=2, bg="#cccccc", relief="sunken")
separator.pack(fill="x", padx=20, pady=(0, 10))

welcome = tk.Label(root, text="Welcome! Choose an option below to get started.", font=("Arial", 12), fg="#555555", bg="#f0f0f0")
welcome.pack(pady=10)

start_button = tk.Button(root, text="Start Chat", width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), command=start_chat)
start_button.pack(pady=5)


#Footer (development version) --> looks profesh i dunno 
# ---------------------------
footer = tk.Label(
    root, text="Version 0.1 - Development Build",
    font=("Arial", 10),
    fg="#777777",
    bg="#f0f0f0"
)

# STARTs
# ---------------------------
root.mainloop()
