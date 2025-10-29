import os
import sys
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

from . import global_state as gs
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BUTTON_RADIUS,
    CRITICAL,
    CRITICAL_HOVER,
    FONT_BOLD,
    FONT_UI,
    GLASS_BG,
    MUTED,
    OUTPUTS_PATH,
    SURFACE_PRIMARY,
    TEXT,
)
from .ui_helpers import update_status


def ensure_user_dirs() -> None:
    if not gs.current_user:
        return
    (Path("history") / gs.current_user).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_PATH / gs.current_user).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_PATH / gs.current_user / ".thumbnails").mkdir(parents=True, exist_ok=True)


def _thumb_path(path: Path) -> Path:
    base = OUTPUTS_PATH / (gs.current_user or "")
    root = base / ".thumbnails" if gs.current_user else OUTPUTS_PATH / ".thumbnails"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{path.stem}_256.png"


def _make_thumbnail(path: Path) -> Path:
    thumb = _thumb_path(path)
    if thumb.exists():
        return thumb
    try:
        with Image.open(path) as image:
            image.thumbnail((256, 256))
            image.save(thumb, "PNG")
        return thumb
    except Exception:
        return path


def _list_images(limit: int | None = None, offset: int = 0) -> list[Path]:
    base = OUTPUTS_PATH / (gs.current_user or "")
    if not base.exists():
        return []
    paths = [
        p for p in base.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is None:
        return paths[offset:]
    return paths[offset : offset + limit]


def _styled_button(parent, text, command, *, danger=False) -> None:
    fg_color = CRITICAL if danger else ACCENT
    hover_color = CRITICAL_HOVER if danger else ACCENT_HOVER
    ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=82,
        corner_radius=BUTTON_RADIUS,
        fg_color=fg_color,
        hover_color=hover_color,
        text_color="white",
        font=FONT_BOLD,
    ).pack(side="left", padx=4)


def refresh_gallery(container) -> None:
    if container is None:
        return
    for child in container.winfo_children():
        child.destroy()
    images = _list_images()
    if not images:
        ctk.CTkLabel(
            container,
            text="No images yet. Generate something to see it here!",
            font=FONT_UI,
            text_color=MUTED,
        ).pack(padx=12, pady=24)
        return
    for path in images:
        thumb_path = _make_thumbnail(path)
        try:
            with Image.open(thumb_path) as image:
                tk_image = ImageTk.PhotoImage(image.copy())
        except Exception:
            continue
        frame = ctk.CTkFrame(
            container,
            fg_color=GLASS_BG,
            corner_radius=BUTTON_RADIUS,
        )
        frame.pack(side="left", padx=10, pady=10)
        label = tk.Label(frame, image=tk_image, bg=SURFACE_PRIMARY)
        label.image = tk_image
        label.pack(padx=8, pady=8)
        name_label = ctk.CTkLabel(frame, text=path.name, text_color=MUTED)
        name_label.pack(pady=(0, 6))
        row_frame = ctk.CTkFrame(frame, fg_color="transparent")
        row_frame.pack(pady=(0, 8))

        def open_image(p: Path = path) -> None:
            resolved = p.resolve()
            if sys.platform == "darwin":
                os.system(f'open "{resolved}"')
                return
            if os.name == "nt":
                os.startfile(str(resolved))
                return
            os.system(f'xdg-open "{resolved}"')

        def reveal_image(p: Path = path) -> None:
            directory = p.resolve().parent
            if sys.platform == "darwin":
                os.system(f'open "{directory}"')
                return
            if os.name == "nt":
                os.startfile(str(directory))
                return
            os.system(f'xdg-open "{directory}"')

        def delete_image(p: Path = path) -> None:
            try:
                from send2trash import send2trash

                send2trash(str(p))
                thumb = _thumb_path(p)
                if thumb.exists():
                    try:
                        thumb.unlink()
                    except Exception:
                        pass
                update_status(f"Moved to trash: {p.name}")
            except Exception:
                if messagebox.askyesno(
                    "Delete permanently?", f"Delete {p.name}? This cannot be undone."
                ):
                    try:
                        p.unlink()
                    except Exception as exc:
                        update_status(f"Delete failed: {exc}")
            refresh_gallery(container)

        _styled_button(row_frame, "Open", open_image)
        _styled_button(row_frame, "Reveal", reveal_image)
        _styled_button(row_frame, "Delete", delete_image, danger=True)
