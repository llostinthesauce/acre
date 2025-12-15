import json
import threading
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import tkinter as tk

from .compat import ensure_tensor_parallel_stub
from platform_utils import is_jetson

from . import global_state as gs
from . import paths
from .constants import (
    ACCENT,
    ACCENT_HOVER,
    BASE_DIR,
    BUTTON_RADIUS,
    FONT_BOLD,
    FONT_UI,
    MUTED,
    OUTPUTS_PATH,
    SUCCESS,
    TEXT,
)
from .ui_helpers import update_status

ensure_tensor_parallel_stub()


def _check_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import Dataset
        return True, None
    except ImportError as e:
        return False, f"Missing dependency: {e}. Install PyTorch, transformers, and datasets."


def _load_dataset(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        records: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue
        return records
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    return []


def _create_demo_dataset() -> List[Dict[str, str]]:
    examples = [
        {"instruction": "What is edge AI?", "response": "Edge AI runs AI algorithms on edge devices like Jetson for real-time processing and privacy."},
        {"instruction": "Explain CUDA.", "response": "CUDA is NVIDIA's parallel computing platform that uses GPUs to accelerate AI workloads."},
        {"instruction": "What makes Jetson special?", "response": "Jetson devices have CUDA cores, Tensor cores, and low power consumption for edge AI."},
    ]
    return examples * 10


JETSON_TRAINING_CONFIG = BASE_DIR / "config" / "jetson_training.json"
JETSON_TRAINING_DOC = BASE_DIR / "jetson" / "README.md"
JETSON_DOC_REFERENCE = "jetson/README.md"


@lru_cache(maxsize=1)
def _load_jetson_training_profile() -> Optional[Dict[str, Any]]:
    if not JETSON_TRAINING_CONFIG.exists():
        return None
    try:
        data = json.loads(JETSON_TRAINING_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _resolve_profile_dataset(profile: Dict[str, Any]) -> Optional[Path]:
    dataset_entry = profile.get("dataset")
    if not dataset_entry or not isinstance(dataset_entry, str):
        return None
    dataset_path = Path(dataset_entry)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path
    if dataset_path.exists() and dataset_path.is_file():
        return dataset_path
    return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def open_training_dialog():
    available, error_msg = _check_training_dependencies()
    if not available:
        messagebox.showerror("Training Unavailable", error_msg)
        return
    
    if not gs.mgr:
        messagebox.showerror("Error", "Model manager not initialized.")
        return

    jetson_profile = _load_jetson_training_profile()
    on_jetson = is_jetson()
    
    models = gs.mgr.list_models()
    models_root = paths.models_dir()
    text_models = [m for m in models if (models_root / m / "config.json").exists()]
    text_models = [m for m in text_models if not any(x in m.lower() for x in ["ocr", "whisper", "tts", "vision", "diffusion", "sd-", "flux", "mlx"])]
    
    if not text_models:
        messagebox.showerror("No Models", "No suitable models found for training.")
        return
    
    window = ctk.CTkToplevel(gs.root)
    window.title("Train Model")
    window.geometry("600x700")
    
    main_frame = ctk.CTkScrollableFrame(window, fg_color="transparent")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    ctk.CTkLabel(main_frame, text="Fine-Tune Model", font=("", 18, "bold"), text_color=TEXT).pack(anchor="w", pady=(0, 10))
    ctk.CTkLabel(main_frame, text="Fine-tune a base model on your dataset (full-model training, no LoRA).", font=FONT_UI, text_color=MUTED, wraplength=560).pack(anchor="w", pady=(0, 20))
    
    model_var = tk.StringVar(value=text_models[0] if text_models else "")
    ctk.CTkLabel(main_frame, text="Base Model:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 5))
    ctk.CTkOptionMenu(main_frame, values=text_models, variable=model_var, font=FONT_UI).pack(anchor="w", fill="x", pady=(0, 15))
    
    dataset_path_var = tk.StringVar()
    dataset_info_var = tk.StringVar(value="No dataset selected")
    if on_jetson and jetson_profile:
        profile_dataset = _resolve_profile_dataset(jetson_profile)
        if profile_dataset:
            dataset_path_var.set(str(profile_dataset))
            try:
                loaded = _load_dataset(profile_dataset)
                dataset_info_var.set(f"Jetson profile dataset loaded ({len(loaded)} examples)")
            except Exception:
                dataset_info_var.set("Jetson profile dataset selected")
        else:
            dataset_info_var.set("Jetson profile dataset unavailable")
    
    ctk.CTkLabel(main_frame, text="Dataset:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 5))
    
    def select_file():
        path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("JSON/JSONL", "*.json *.jsonl *.jsonlines")])
        if path:
            try:
                data = _load_dataset(Path(path))
                dataset_path_var.set(path)
                dataset_info_var.set(f"Loaded {len(data)} examples")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    row = ctk.CTkFrame(main_frame, fg_color="transparent")
    row.pack(fill="x", pady=(0, 5))
    ctk.CTkButton(row, text="Select File", command=select_file, width=120).pack(side="left", padx=(0, 10))
    ctk.CTkButton(row, text="Use Demo", command=lambda: [dataset_path_var.set("__demo__"), dataset_info_var.set("Using demo dataset")], width=120).pack(side="left")
    ctk.CTkLabel(main_frame, textvariable=dataset_info_var, font=FONT_UI, text_color=MUTED).pack(anchor="w", pady=(0, 15))

    if on_jetson:
        profile_note = ""
        if jetson_profile:
            note = jetson_profile.get("notes")
            profile_note = str(note).strip() if isinstance(note, str) and note.strip() else ""
        if not profile_note:
            profile_note = "Jetson profile engaged: defaults favor default batch=1, gradient accumulation, and checkpointing."
        ctk.CTkLabel(
            main_frame,
            text=profile_note,
            font=FONT_UI,
            text_color=ACCENT,
            wraplength=560,
        ).pack(anchor="w", pady=(0, 8))
        doc_label = f"Refer to {JETSON_DOC_REFERENCE} and run scripts/jetson_torch_install.py for dependency steps."
        ctk.CTkLabel(
            main_frame,
            text=doc_label,
            font=FONT_UI,
            text_color=MUTED,
            wraplength=560,
        ).pack(anchor="w", pady=(0, 12))
    
    ctk.CTkLabel(main_frame, text="Parameters:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 10))
    
    profile_defaults = jetson_profile.get("defaults", {}) if jetson_profile else {}
    epochs_default = _as_int(profile_defaults.get("epochs"), 3)
    lr_default = _as_float(profile_defaults.get("learning_rate"), 0.0002)
    batch_default = _as_int(profile_defaults.get("batch_size"), 1)
    epochs_var = tk.IntVar(value=epochs_default)
    lr_var = tk.StringVar(value=str(lr_default))
    batch_var = tk.IntVar(value=batch_default)
    output_var = tk.StringVar()
    
    for label, var in [("Epochs", epochs_var), ("Learning Rate", lr_var), ("Batch Size", batch_var), ("Output Name", output_var)]:
        row = ctk.CTkFrame(main_frame, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(row, text=f"{label}:", font=FONT_UI, width=120).pack(side="left")
        if isinstance(var, tk.IntVar):
            ctk.CTkEntry(row, textvariable=var, width=100).pack(side="left", padx=(10, 0))
        else:
            ctk.CTkEntry(row, textvariable=var, width=200).pack(side="left", padx=(10, 0))
    
    progress_label = ctk.CTkLabel(main_frame, text="Status: Ready", font=FONT_UI, text_color=TEXT)
    progress_label.pack(anchor="w", pady=(15, 5))
    progress_bar = ctk.CTkProgressBar(main_frame, width=560)
    progress_bar.pack(fill="x", pady=(0, 5))
    progress_text = ctk.CTkTextbox(main_frame, height=100, width=560, font=("Courier", 10))
    progress_text.pack(fill="x", pady=(0, 15))
    
    training_active = {"value": False}
    
    def update_progress(msg, progress):
        progress_label.configure(text=f"Status: {msg}")
        progress_bar.set(max(0, min(1, progress)))
        progress_text.insert("end", f"{msg}\n")
        progress_text.see("end")
        progress_text.update()
    
    def train_complete(output_name):
        training_active["value"] = False
        start_btn.configure(state="normal")
        progress_label.configure(text=f"Complete! Model: {output_name}")
        from .models import refresh_list
        refresh_list()
        update_status(f"Training complete: {output_name}")
        messagebox.showinfo("Complete", f"Model saved as: {output_name}")
    
    def train_failed(error_msg):
        training_active["value"] = False
        start_btn.configure(state="normal")
        progress_label.configure(text="Failed")
        messagebox.showerror("Failed", f"Error: {error_msg}")
    
    def start_training():
        if not dataset_path_var.get():
            messagebox.showerror("Error", "Please select a dataset.")
            return
        
        try:
            epochs = int(epochs_var.get())
            learning_rate = float(lr_var.get())
            batch_size = int(batch_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return
        
        base_model = model_var.get()
        output_name = output_var.get().strip() or f"{base_model}-ft"
        
        if dataset_path_var.get() == "__demo__":
            training_data = _create_demo_dataset()
        else:
            try:
                training_data = _load_dataset(Path(dataset_path_var.get()))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
                return
        
        training_active["value"] = True
        start_btn.configure(state="disabled")
        progress_text.delete("1.0", "end")
        progress_bar.set(0)
        
        def train_thread():
            try:
                _run_training(
                    base_model,
                    training_data,
                    output_name,
                    epochs,
                    learning_rate,
                    batch_size,
                    jetson_profile if on_jetson else None,
                    update_progress,
                )
                window.after(0, lambda: train_complete(output_name))
            except Exception as e:
                error_msg = str(e)
                window.after(0, lambda: train_failed(error_msg))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.pack(fill="x", pady=(10, 0))
    start_btn = ctk.CTkButton(button_frame, text="Start Training", command=start_training, fg_color=SUCCESS, width=150)
    start_btn.pack(side="left", padx=(0, 10))


def _run_training(
    base_model: str,
    training_data: List[Dict[str, str]],
    output_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    jetson_profile: Optional[Dict[str, Any]],
    progress_callback,
):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )

    device_pref = "auto"
    if gs.mgr:
        device_pref = getattr(gs.mgr, "_device_pref", "auto") or "auto"
    device = _determine_training_device(device_pref, torch)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    progress_callback("Loading model...", 0.1)
    models_root = paths.models_dir()
    model_path = models_root / base_model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)

    progress_callback("Preparing dataset...", 0.2)
    texts = [
        f"### Instruction:\n{ex.get('instruction', '')}\n\n### Response:\n{ex.get('response', '')}\n"
        for ex in training_data
    ]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    if jetson_profile:
        progress_callback("Jetson profile: checkpointing + accumulation active.", 0.25)
    progress_callback("Starting training...", 0.3)

    output_dir = models_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_swap_directory(jetson_profile)

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "fp16": device == "cuda",
        "logging_steps": 10,
        "save_steps": 1000,
        "save_total_limit": 2,
        "report_to": None,
    }
    training_kwargs = _merge_jetson_training_kwargs(training_kwargs, jetson_profile)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    progress_callback("Saving model...", 0.9)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    for file in ["config.json", "generation_config.json", "tokenizer.model", "vocab.json", "merges.txt"]:
        src = model_path / file
        dst = output_dir / file
        if src.exists() and not dst.exists():
            import shutil

            shutil.copy2(src, dst)

    progress_callback("Complete!", 1.0)


def _merge_jetson_training_kwargs(
    kwargs: Dict[str, Any], profile: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not profile:
        return kwargs
    overrides = profile.get("training_arguments")
    if not isinstance(overrides, dict):
        return kwargs
    blocked = {
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
        "learning_rate",
        "report_to",
    }
    for key, value in overrides.items():
        if key in blocked:
            continue
        kwargs[key] = value
    return kwargs


def _ensure_swap_directory(profile: Optional[Dict[str, Any]]) -> None:
    if not profile:
        return
    swap_dir = profile.get("swap_directory")
    if not swap_dir or not isinstance(swap_dir, str):
        return
    path = Path(swap_dir)
    if not path.is_absolute():
        path = BASE_DIR / path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _determine_training_device(pref: str, torch_module) -> str:
    if is_jetson():
        return "cpu"
    candidate = (pref or "auto").lower()
    if candidate in ("auto", "cuda"):
        try:
            if torch_module.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if candidate == "mps":
        try:
            backend = getattr(torch_module, "backends", None)
            if backend and getattr(backend, "mps", None) and backend.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"
