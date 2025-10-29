import tkinter.font as tkfont

from . import global_state as gs
from .constants import BASE_UI, FONT_FAMILY


def apply_native_font_scale(scale: float) -> None:
    size = max(10, int(BASE_UI * scale))
    if gs.native_tk_font is None:
        gs.native_tk_font = tkfont.Font(family=FONT_FAMILY, size=size)
    else:
        gs.native_tk_font.configure(size=size)
    if gs.listbox is not None:
        gs.listbox.configure(font=gs.native_tk_font)
    if gs.entry is not None:
        gs.entry.configure(font=gs.native_tk_font)
    if gs.chat_history is not None:
        gs.chat_history.configure(font=(FONT_FAMILY, max(10, int(BASE_UI * scale))))


def update_status(message: str) -> None:
    if gs.status:
        try:
            gs.status.configure(state="normal")
            gs.status.delete("1.0", "end")
            text = (message or "").strip()
            if text:
                gs.status.insert("1.0", text)
            gs.status.configure(state="disabled")
            gs.status.see("1.0")
        except AttributeError:
            gs.status.configure(text=message, wraplength=320, justify="left")


def update_logo_visibility() -> None:
    if gs.logo_label is None or gs.chat_history is None:
        return
    content = gs.chat_history.get("1.0", "end-1c").strip()
    try:
        if content:
            gs.logo_label.place_forget()
        else:
            gs.logo_label.place(relx=0.5, rely=0.4, anchor="center")
            gs.logo_label.lift()
    except Exception:
        pass
