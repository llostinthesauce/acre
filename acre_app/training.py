import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List

import customtkinter as ctk
import tkinter as tk

from . import global_state as gs
from .constants import ACCENT_HOVER, BUTTON_RADIUS, FONT_BOLD, FONT_UI, MODELS_PATH, MUTED, SUCCESS, TEXT
from .ui_helpers import update_status


def _check_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        return True, None
    except ImportError as e:
        return False, f"Missing dependency: {e}. Install PyTorch, transformers, peft, and datasets."


def _load_dataset(path: Path) -> List[Dict[str, str]]:
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


def open_training_dialog():
    available, error_msg = _check_training_dependencies()
    if not available:
        messagebox.showerror("Training Unavailable", error_msg)
        return
    
    if not gs.mgr:
        messagebox.showerror("Error", "Model manager not initialized.")
        return
    
    models = gs.mgr.list_models()
    text_models = [m for m in models if (Path(MODELS_PATH) / m / "config.json").exists()]
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
    ctk.CTkLabel(main_frame, text="Fine-tune a base model on your dataset.", font=FONT_UI, text_color=MUTED, wraplength=560).pack(anchor="w", pady=(0, 20))
    
    model_var = tk.StringVar(value=text_models[0] if text_models else "")
    ctk.CTkLabel(main_frame, text="Base Model:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 5))
    ctk.CTkOptionMenu(main_frame, values=text_models, variable=model_var, font=FONT_UI).pack(anchor="w", fill="x", pady=(0, 15))
    
    dataset_path_var = tk.StringVar()
    dataset_info_var = tk.StringVar(value="No dataset selected")
    
    ctk.CTkLabel(main_frame, text="Dataset:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 5))
    
    def select_file():
        path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("JSON", "*.json")])
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
    
    ctk.CTkLabel(main_frame, text="Parameters:", font=FONT_BOLD, text_color=TEXT).pack(anchor="w", pady=(0, 10))
    
    epochs_var = tk.IntVar(value=3)
    lr_var = tk.StringVar(value="2e-4")
    batch_var = tk.IntVar(value=1)
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
        output_name = output_var.get().strip() or f"{base_model}-lora"
        
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
                _run_training(base_model, training_data, output_name, epochs, learning_rate, batch_size, update_progress)
                window.after(0, lambda: train_complete(output_name))
            except Exception as e:
                window.after(0, lambda: train_failed(str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.pack(fill="x", pady=(10, 0))
    start_btn = ctk.CTkButton(button_frame, text="Start Training", command=start_training, fg_color=SUCCESS, width=150)
    start_btn.pack(side="left", padx=(0, 10))


def _run_training(base_model: str, training_data: List[Dict[str, str]], output_name: str, epochs: int, learning_rate: float, batch_size: int, progress_callback):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    device = "cuda" if (torch.cuda.is_available() and gs.mgr and gs.mgr._device_pref == "cuda") else "cpu"
    
    progress_callback("Loading model...", 0.1)
    model_path = MODELS_PATH / base_model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)
    
    progress_callback("Applying LoRA...", 0.2)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    progress_callback("Preparing dataset...", 0.3)
    texts = [f"### Instruction:\n{ex.get('instruction', '')}\n\n### Response:\n{ex.get('response', '')}\n" for ex in training_data]
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    progress_callback("Starting training...", 0.4)
    output_dir = MODELS_PATH / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=device == "cuda",
            logging_steps=10,
            save_steps=1000,
            save_total_limit=2,
            report_to=None,
        ),
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
