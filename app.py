import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Optional

from model_manager import model_manager
from security import ChatHistoryStore, SecurityManager


APP_BG = "#070a12"
GLASS_BG = "#101722"
PANEL_BG = "#141d2d"
ACCENT = "#5a8dee"
ACCENT_HOVER = "#6d9bff"
BUTTON_BG = "#dbe3ff"
BUTTON_HOVER = "#c6d7ff"
TEXT_COLOR = "#e6ecfc"
MUTED_TEXT = "#8b98b8"
ERROR_COLOR = "#ff6b6b"


class AuthDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, security: SecurityManager):
        super().__init__(parent)
        self.security = security
        self.success = False
        self.mode = "setup" if not security.is_password_set() else "login"

        self.configure(bg=GLASS_BG)
        self.title("Unlock Offline LLM Switcher")
        self.geometry("380x260")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build_ui()
        self.update_idletasks()
        self._center_on_screen()
        try:
            self.deiconify()
        except Exception:
            pass
        self.lift()
        try:
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass
        self.password_entry.focus_set()
        parent.wait_window(self)

    def _build_ui(self) -> None:
        title = "Welcome! Set a password." if self.mode == "setup" else "Enter your password"
        tk.Label(
            self,
            text=title,
            fg=TEXT_COLOR,
            bg=GLASS_BG,
            font=("Segoe UI Semibold", 13),
        ).pack(pady=(18, 4))

        subtitle_text = (
            "This password encrypts chat history. Store it safely—there is no reset."
            if self.mode == "setup"
            else "Data stays encrypted until you unlock the app."
        )
        tk.Label(
            self,
            text=subtitle_text,
            fg=MUTED_TEXT,
            bg=GLASS_BG,
            wraplength=300,
            justify="center",
            font=("Segoe UI", 10),
        ).pack(pady=(0, 18))

        form = tk.Frame(self, bg=GLASS_BG)
        form.pack(expand=True)

        self.password_entry = self._build_entry(form, "Password")
        if self.mode == "setup":
            self.confirm_entry = self._build_entry(form, "Confirm Password")
        else:
            self.confirm_entry = None

        self.error_var = tk.StringVar(value="")
        tk.Label(
            self,
            textvariable=self.error_var,
            fg=ERROR_COLOR,
            bg=GLASS_BG,
            font=("Segoe UI", 9),
        ).pack(pady=(4, 0))

        button_text = "Save Password" if self.mode == "setup" else "Unlock"
        action_btn = tk.Button(
            self,
            text=button_text,
            command=self._submit,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=10,
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
        )
        action_btn.pack(pady=(16, 6))
        action_btn.bind("<Enter>", lambda _e: action_btn.config(bg=ACCENT_HOVER))
        action_btn.bind("<Leave>", lambda _e: action_btn.config(bg=ACCENT))

        self.bind("<Return>", lambda _e: self._submit())

    def _build_entry(self, parent: tk.Frame, label: str) -> tk.Entry:
        wrapper = tk.Frame(parent, bg=GLASS_BG)
        wrapper.pack(fill=tk.X, padx=26, pady=6)

        tk.Label(
            wrapper,
            text=label,
            fg=TEXT_COLOR,
            bg=GLASS_BG,
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill=tk.X)

        entry = tk.Entry(
            wrapper,
            show="•",
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#243044",
            highlightcolor=ACCENT,
            font=("Segoe UI", 11),
            justify="left",
        )
        entry.pack(fill=tk.X, ipady=6, pady=(4, 0))
        return entry

    def _submit(self) -> None:
        password = self.password_entry.get().strip()
        if not password:
            self.error_var.set("Password cannot be empty.")
            return

        if self.mode == "setup":
            confirm = (self.confirm_entry.get().strip() if self.confirm_entry else "")
            if password != confirm:
                self.error_var.set("Passwords do not match.")
                return
            self.security.set_password(password)
            self.success = True
            self.destroy()
        else:
            if self.security.verify_password(password):
                self.success = True
                self.destroy()
            else:
                self.error_var.set("Incorrect password. Try again.")

    def _cancel(self) -> None:
        self.success = False
        self.destroy()

    def _center_on_screen(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.geometry(f"{width}x{height}+{x}+{y}")


class SwitcherApp:
    def __init__(self) -> None:
        self.security = SecurityManager()
        self.manager = model_manager()
        self.chat_store: Optional[ChatHistoryStore] = None

        self.root: Optional[tk.Tk] = None
        self.listbox: Optional[tk.Listbox] = None
        self.chat_history: Optional[scrolledtext.ScrolledText] = None
        self.entry: Optional[tk.Text] = None
        self.load_button: Optional[tk.Button] = None
        self.send_button: Optional[tk.Button] = None
        self.clear_button: Optional[tk.Button] = None
        self.history_button: Optional[tk.Button] = None
        self.status_label: Optional[tk.Label] = None
        self.meta_var: Optional[tk.StringVar] = None
        self.status_var: Optional[tk.StringVar] = None
        self._meta_text = "No model selected"
        self._status_text = "Locked until a model is loaded."

        self.current_model_name: Optional[str] = None
        self.conversation: list[dict[str, str]] = []
        self.conversation_lock = threading.Lock()

    def _update_meta(self, text: str) -> None:
        self._meta_text = text
        if self.meta_var:
            self.meta_var.set(text)

    def _update_status(self, text: str) -> None:
        self._status_text = text
        if self.status_var:
            self.status_var.set(text)

    # ------------------------------------------------------------------ Lifecycle
    def run(self) -> None:
        self.root = tk.Tk()
        self.meta_var = tk.StringVar(master=self.root, value=self._meta_text)
        self.status_var = tk.StringVar(master=self.root, value=self._status_text)

        auth = AuthDialog(self.root, self.security)
        if not auth.success:
            self.root.destroy()
            return

        self.root.withdraw()

        self.chat_store = ChatHistoryStore(security=self.security)
        self._configure_root()
        self._build_ui()
        self.refresh_model_list()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.deiconify()
        self.root.mainloop()

    def _configure_root(self) -> None:
        if not self.root:
            return
        self.root.title("Offline LLM Switcher — Secure Edition")
        self.root.geometry("980x720")
        self.root.minsize(900, 620)
        self.root.configure(bg=APP_BG)
        try:
            self.root.attributes("-alpha", 0.96)
        except Exception:
            pass  # attribute not available on some platforms

    # ------------------------------------------------------------------ UI helpers
    def _build_ui(self) -> None:
        if not self.root:
            return

        container = tk.Frame(self.root, bg=APP_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=28, pady=28)

        glass = tk.Frame(
            container,
            bg=GLASS_BG,
            highlightthickness=1,
            highlightbackground="#1e2738",
            bd=0,
        )
        glass.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(glass, bg=PANEL_BG, width=280, highlightthickness=0)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        sidebar.pack_propagate(False)

        chat_area = tk.Frame(glass, bg=PANEL_BG)
        chat_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Sidebar header
        tk.Label(
            sidebar,
            text="Models",
            fg=TEXT_COLOR,
            bg=PANEL_BG,
            font=("Segoe UI Semibold", 14),
            anchor="w",
            pady=12,
        ).pack(fill=tk.X, padx=18)

        list_frame = tk.Frame(sidebar, bg=PANEL_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=18)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame,
            bg="#101724",
            fg=TEXT_COLOR,
            selectbackground="#253957",
            selectforeground=TEXT_COLOR,
            activestyle="none",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground="#1f2737",
            highlightcolor=ACCENT,
            exportselection=False,
            font=("Segoe UI", 11),
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        button_bar = tk.Frame(sidebar, bg=PANEL_BG)
        button_bar.pack(fill=tk.X, padx=18, pady=12)

        refresh_btn = self._make_button(
            button_bar, "Refresh", self.refresh_model_list, accent=False
        )
        refresh_btn.pack(fill=tk.X, pady=4)

        self.load_button = self._make_button(
            button_bar, "Load Model", self.pick_model, accent=True
        )
        self.load_button.pack(fill=tk.X, pady=4)

        add_btn = self._make_button(
            button_bar, "Add Model", self.add_model, accent=False
        )
        add_btn.pack(fill=tk.X, pady=4)

        self.history_button = self._make_button(
            button_bar, "Clear History", self.clear_chat, accent=False
        )
        self.history_button.pack(fill=tk.X, pady=4)

        tk.Label(
            sidebar,
            textvariable=self.meta_var,
            fg=MUTED_TEXT,
            bg=PANEL_BG,
            wraplength=220,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(fill=tk.X, padx=18, pady=(6, 12))

        # Chat header
        header = tk.Frame(chat_area, bg=PANEL_BG)
        header.pack(fill=tk.X, padx=24, pady=(20, 8))

        tk.Label(
            header,
            text="Secure Chat Workspace",
            fg=TEXT_COLOR,
            bg=PANEL_BG,
            font=("Segoe UI Semibold", 16),
        ).pack(side=tk.LEFT)

        self.clear_button = self._make_button(
            header, "Clear Chat", self.clear_chat, accent=True
        )
        self.clear_button.pack(side=tk.RIGHT)

        # Chat history
        history_frame = tk.Frame(chat_area, bg=PANEL_BG)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 12))

        self.chat_history = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#0d1320",
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 11),
            padx=18,
            pady=16,
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        self.chat_history.tag_configure("user_name", foreground="#82aaff", font=("Segoe UI Semibold", 11))
        self.chat_history.tag_configure("assistant_name", foreground="#f6c177", font=("Segoe UI Semibold", 11))
        self.chat_history.tag_configure("user", foreground="#dbe3ff", font=("Segoe UI", 11))
        self.chat_history.tag_configure("assistant", foreground="#f7f8ff", font=("Segoe UI", 11))

        # Input area
        input_frame = tk.Frame(chat_area, bg=PANEL_BG)
        input_frame.pack(fill=tk.X, padx=24, pady=(0, 18))

        self.entry = tk.Text(
            input_frame,
            height=4,
            bg="#101724",
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT,
            bd=0,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
        )
        self.entry.pack(fill=tk.X, pady=(0, 8))
        self.entry.bind("<Return>", self._handle_return)

        controls = tk.Frame(input_frame, bg=PANEL_BG)
        controls.pack(fill=tk.X)

        hint = tk.Label(
            controls,
            text="Shift + Enter for newline",
            fg=MUTED_TEXT,
            bg=PANEL_BG,
            font=("Segoe UI", 9),
        )
        hint.pack(side=tk.LEFT)

        self.send_button = self._make_button(
            controls, "Send", self.run_prompt, accent=True
        )
        self.send_button.pack(side=tk.RIGHT)

        # Status bar
        status_frame = tk.Frame(chat_area, bg=PANEL_BG)
        status_frame.pack(fill=tk.X, padx=24, pady=(0, 20))

        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            fg=MUTED_TEXT,
            bg=PANEL_BG,
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(fill=tk.X)

        self._set_interaction_enabled(False)

    def _make_button(self, parent: tk.Widget, text: str, command, accent: bool) -> tk.Button:
        if accent:
            bg = ACCENT
            hover = ACCENT_HOVER
        else:
            bg = BUTTON_BG
            hover = BUTTON_HOVER
        fg = APP_BG

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=APP_BG,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda _e: btn.config(bg=bg))
        return btn

    # ------------------------------------------------------------------ Model ops
    def refresh_model_list(self) -> None:
        if not self.listbox:
            return
        models = self.manager.list_models()
        self.listbox.delete(0, tk.END)
        for model in models:
            self.listbox.insert(tk.END, model)

        if self.current_model_name and self.current_model_name in models:
            index = models.index(self.current_model_name)
            self.listbox.selection_set(index)
            self.listbox.activate(index)
            self.listbox.see(index)

    def pick_model(self) -> None:
        if not self.listbox:
            return
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Select Model", "Please choose a model from the list.")
            return
        name = self.listbox.get(selection[0])

        if self.load_button:
            self.load_button.config(state=tk.DISABLED)
        loaded = self.manager.load_model(name)
        if self.load_button:
            self.load_button.config(state=tk.NORMAL)

        if loaded:
            self.current_model_name = name
            config = self.manager.describe_model(name) or {}
            descriptor = config.get("model_type", "unknown")
            backend = config.get("backend", "ctransformers")
            self._update_meta(f"{name}\nbackend: {backend}\ntype: {descriptor}")
            self._update_status(f"Model '{name}' loaded. Ready for prompts.")
            self._set_interaction_enabled(True)
            self._load_chat_history(name)
            if self.listbox:
                self.listbox.selection_clear(0, tk.END)
                index = self.listbox.get(0, tk.END).index(name)
                self.listbox.selection_set(index)
                self.listbox.activate(index)
                self.listbox.see(index)
        else:
            messagebox.showerror("Load Failed", f"Unable to load '{name}'. Check the logs.")
            self._update_status("Failed to load model. See console for details.")
            self._set_interaction_enabled(False)

    def add_model(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Pick a model file",
            filetypes=[("Model files", "*.gguf *.bin *.safetensors")],
        )
        if not file_path:
            return

        dest_dir = "models"
        base_name = os.path.basename(file_path)
        extension = os.path.splitext(base_name)[1].lower()
        removal_target: Optional[str] = None

        if extension == ".safetensors":
            source_dir = os.path.dirname(file_path.rstrip(os.sep))
            folder_name = os.path.basename(source_dir)
            dest_path = os.path.join(dest_dir, folder_name)
            unique_path = dest_path
            counter = 1
            while os.path.exists(unique_path):
                unique_path = f"{dest_path}_{counter}"
                counter += 1

            try:
                shutil.copytree(source_dir, unique_path)
            except Exception as exc:
                messagebox.showerror("Copy failed", f"Could not copy model directory:\n{exc}")
                return

            removal_target = unique_path
            stored_path = os.path.basename(unique_path)
            default_name = stored_path
        else:
            dest_path = os.path.join(dest_dir, base_name)
            unique_path = dest_path
            stem, ext = os.path.splitext(base_name)
            counter = 1
            while os.path.exists(unique_path):
                unique_path = os.path.join(dest_dir, f"{stem}_{counter}{ext}")
                counter += 1

            try:
                shutil.copy2(file_path, unique_path)
            except Exception as exc:
                messagebox.showerror("Copy failed", f"Could not copy model file:\n{exc}")
                return

            removal_target = unique_path
            stored_path = os.path.basename(unique_path)
            default_name = os.path.splitext(stored_path)[0]
        display_name = self._prompt_text("Model Name", "Enter display name for this model:", default_name)
        if not display_name:
            self._remove_path(removal_target)
            return

        model_type_hint = self._prompt_text(
            "Model Type",
            "Enter the base architecture for this model.\n"
            "(examples: llama, qwen2, deepseek, mistral, phi).\n"
            "Leave blank to auto-detect from the filename.",
            "",
        )

        suggested_backend = "transformers" if extension == ".safetensors" else "ctransformers"
        backend_choice = self._prompt_text(
            "Execution Backend",
            "Choose runtime backend:\n"
            "- ctransformers (GGUF/bin quantized models)\n"
            "- transformers (safetensors Hugging Face format; requires torch + transformers)\n"
            "Type your choice exactly.",
            suggested_backend,
        )
        if backend_choice is None:
            self._remove_path(removal_target)
            return

        backend_choice = backend_choice.strip().lower()
        if backend_choice not in {"ctransformers", "transformers"}:
            self._remove_path(removal_target)
            messagebox.showerror("Invalid backend", "Backend must be 'ctransformers' or 'transformers'.")
            return

        try:
            self.manager.register_model(
                name=display_name.strip(),
                path=stored_path,
                model_type=(model_type_hint or default_name),
                backend=backend_choice,
            )
            messagebox.showinfo("Model Added", f"Registered offline model:\n{display_name}")
        except Exception as exc:
            self._remove_path(removal_target)
            messagebox.showerror("Registration failed", f"Could not register model:\n{exc}")
            return

        self.refresh_model_list()

    def _remove_path(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _prompt_text(self, title: str, message: str, initial: str = "") -> Optional[str]:
        if not self.root:
            return None
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=GLASS_BG)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text=message,
            fg=TEXT_COLOR,
            bg=GLASS_BG,
            wraplength=360,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(padx=20, pady=(20, 10))

        var = tk.StringVar(value=initial)
        entry = tk.Entry(
            dialog,
            textvariable=var,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#1f2738",
            highlightcolor=ACCENT,
            font=("Segoe UI", 11),
        )
        entry.pack(fill=tk.X, padx=20, pady=(0, 20), ipady=6)
        entry.focus_set()

        buttons = tk.Frame(dialog, bg=GLASS_BG)
        buttons.pack(pady=(0, 18))

        result = {"value": None}

        def submit():
            result["value"] = var.get().strip()
            dialog.destroy()

        def cancel():
            result["value"] = None
            dialog.destroy()

        ok_btn = self._make_button(buttons, "OK", submit, accent=True)
        ok_btn.pack(side=tk.LEFT, padx=10)
        cancel_btn = self._make_button(buttons, "Cancel", cancel, accent=False)
        cancel_btn.pack(side=tk.LEFT, padx=10)

        dialog.bind("<Return>", lambda _e: submit())
        dialog.bind("<Escape>", lambda _e: cancel())

        self.root.wait_window(dialog)
        return result["value"]

    # ------------------------------------------------------------------ Chat flow
    def run_prompt(self) -> None:
        if not self.entry or not self.send_button:
            return
        if not self.current_model_name:
            messagebox.showinfo("Model required", "Load a model before sending prompts.")
            return

        text = self.entry.get("1.0", tk.END).strip()
        if not text:
            return

        self.entry.delete("1.0", tk.END)
        self.entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

        self._append_message("user", text)
        with self.conversation_lock:
            self.conversation.append({"role": "user", "content": text})
        self._persist_conversation()

        self._update_status("Generating response...")

        def task(prompt: str) -> None:
            try:
                output_text = self.manager.generate(prompt)
            except Exception as exc:
                output_text = f"Error: {exc}"

            with self.conversation_lock:
                self.conversation.append({"role": "assistant", "content": output_text})
            self._persist_conversation()

            if self.chat_history:
                self.chat_history.after(0, lambda: self._finalize_generation(output_text))

        threading.Thread(target=task, args=(text,), daemon=True).start()

    def _finalize_generation(self, response: str) -> None:
        if not self.entry or not self.send_button:
            return

        self._append_message("assistant", response)
        self.entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.entry.focus_set()

        self._update_status("Ready for the next prompt.")

    def _append_message(self, role: str, content: str) -> None:
        if not self.chat_history:
            return
        self.chat_history.config(state=tk.NORMAL)
        self._insert_message(role, content)
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def _insert_message(self, role: str, content: str) -> None:
        if not self.chat_history:
            return
        label = "You" if role == "user" else "Assistant"
        name_tag = "user_name" if role == "user" else "assistant_name"
        text_tag = "user" if role == "user" else "assistant"
        self.chat_history.insert(tk.END, f"{label}: ", name_tag)
        self.chat_history.insert(tk.END, f"{content}\n\n", text_tag)

    def _render_full_conversation(self) -> None:
        if not self.chat_history:
            return
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        for message in self.conversation:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            self._insert_message(role, content)
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def _handle_return(self, event) -> Optional[str]:
        if event.state & 0x0001:  # Shift key
            return None
        self.run_prompt()
        return "break"

    def clear_chat(self) -> None:
        if not self.current_model_name:
            messagebox.showinfo("No Model", "Load a model before clearing chat.")
            return
        if not self.chat_store:
            return
        if not messagebox.askyesno(
            "Clear Chat History",
            f"Erase chat history for '{self.current_model_name}'?",
            icon="warning",
        ):
            return
        with self.conversation_lock:
            self.conversation.clear()
        self.chat_store.clear(self.current_model_name)
        self._render_full_conversation()
        self._update_status("Chat cleared for this model.")

    def _persist_conversation(self) -> None:
        if not self.current_model_name or not self.chat_store:
            return
        with self.conversation_lock:
            snapshot = list(self.conversation)
        self.chat_store.save(self.current_model_name, snapshot)

    def _load_chat_history(self, model_name: str) -> None:
        if not self.chat_store:
            return
        try:
            history = self.chat_store.load(model_name)
        except ValueError:
            messagebox.showwarning(
                "History Error",
                f"Encrypted chat history for '{model_name}' could not be decrypted. The log will be reset.",
            )
            if self.chat_store:
                self.chat_store.clear(model_name)
            history = []
        with self.conversation_lock:
            self.conversation = history
        if self.conversation:
            self._render_full_conversation()
            self._update_status("Loaded saved conversation for this model.")
        else:
            if self.chat_history:
                self.chat_history.config(state=tk.NORMAL)
                self.chat_history.delete("1.0", tk.END)
                self.chat_history.config(state=tk.DISABLED)
            self._update_status("New session started for this model.")

    def _set_interaction_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        if self.entry:
            self.entry.config(state=state)
        if self.send_button:
            self.send_button.config(state=state)
        if self.clear_button:
            self.clear_button.config(state=state)
        if self.history_button:
            self.history_button.config(state=state)

    # ------------------------------------------------------------------ Shutdown
    def on_close(self) -> None:
        if self.current_model_name:
            self._persist_conversation()
        if self.root:
            self.root.destroy()


if __name__ == "__main__":
    app = SwitcherApp()
    app.run()
