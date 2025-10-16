import tkinter as tk
from tkinter import messagebox

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



# BIG BIG DISCLAIMERRRR
# ---------------------------
messagebox.showinfo(
    "Disclaimer",
    "Models are offline and may not always provide accurate answers.\n"
    "Responses are based on the data available at the time the model was trained."
)

# Home Screen Layout [ROUGH]
# ---------------------------
header = tk.Label(root, text="ACRE LLM Switchboard", font=("Arial", 18, "bold"))
header.pack(pady=20)

welcome = tk.Label(root, text="Welcome! Choose an option below to get started.", font=("Arial", 12))
welcome.pack(pady=10)

start_button = tk.Button(root, text="Start Chat", width=20, command=start_chat)
start_button.pack(pady=5)


# START
# ---------------------------
root.mainloop()
