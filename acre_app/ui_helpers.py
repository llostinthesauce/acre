import tkinter as tk
import tkinter.font as tkfont

from . import global_state as gs
from . import constants as c
from .constants import BASE_UI, FONT_FAMILY

def apply_native_font_scale(scale: float) -> None:
    size = max(10, int(BASE_UI * scale))
    try:
        if gs.native_tk_font is None:
            gs.native_tk_font = tkfont.Font(family=FONT_FAMILY, size=size)
        else:
            gs.native_tk_font.configure(size=size)
    except Exception:
        try:
            gs.native_tk_font = tkfont.Font(family="TkDefaultFont", size=size)
        except Exception:
            gs.native_tk_font = tkfont.Font(size=size)
    
    if gs.listbox is not None:
        try:
            gs.listbox.configure(font=gs.native_tk_font)
        except Exception:
            pass
    if gs.entry is not None:
        try:
            gs.entry.configure(font=gs.native_tk_font)
        except Exception:
            pass
    if gs.chat_history is not None:
        try:
            gs.chat_history.configure(font=(FONT_FAMILY, max(10, int(BASE_UI * scale))))
        except Exception:
            try:
                gs.chat_history.configure(font=("TkDefaultFont", max(10, int(BASE_UI * scale))))
            except Exception:
                pass


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

def _safe_config(widget, **kwargs):
    if widget is None:
        return
    try:
        widget.configure(**kwargs)
    except Exception:
        pass


def recolor_whole_app(root: tk.Misc):
    """
    Re-apply theme colors to the few plain Tk widgets we use.
    Leave CustomTkinter widgets alone so CTk handles its own styling.
    """
    if root is None:
        return

    try:
        try:
            root.configure(fg_color=c.BG_GRAD_TOP)
        except Exception:
            root.configure(bg=c.BG_GRAD_TOP)
    except Exception:
        pass

    if gs.listbox is not None:
        try:
            gs.listbox.configure(
                bg=c.BG_LIST,
                fg=c.TEXT,
                selectbackground=c.HL_LIST,
                selectforeground=c.TEXT,
                highlightthickness=0,
                bd=0,
                relief="flat",
            )
        except Exception:
            pass

    if gs.entry is not None:
        try:
            gs.entry.configure(
                bg=c.CONTROL_BG,
                fg=c.MUTED,
                insertbackground=c.TEXT,
                highlightthickness=0,
                bd=0,
                relief="flat",
            )
        except Exception:
            pass

    if gs.logo_label is not None:
        try:
            gs.logo_label.configure(bg=c.SURFACE_PRIMARY)
        except Exception:
            pass

    try:
        root.update_idletasks()
    except Exception:
        pass
