from __future__ import annotations

from pathlib import Path
from typing import Iterable


class DocumentReadError(RuntimeError):
    pass


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise DocumentReadError("PyMuPDF is required to read PDF files.") from exc
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise DocumentReadError(f"Unable to open PDF: {exc}") from exc
    texts: list[str] = []
    try:
        for page in doc:
            texts.append(page.get_text())
    finally:
        doc.close()
    text = "\n".join(chunk.strip() for chunk in texts if chunk.strip())
    if not text:
        raise DocumentReadError("No extractable text found in this PDF.")
    return text


def _read_text_file(path: Path) -> str:
    try:
        data = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        data = path.read_text(encoding="latin-1")
    except Exception as exc:
        raise DocumentReadError(f"Unable to read file: {exc}") from exc
    cleaned = data.strip()
    if not cleaned:
        raise DocumentReadError("File appears to be empty.")
    return cleaned


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in {".txt", ".md", ".markdown"}:
        return _read_text_file(path)
    raise DocumentReadError(f"Unsupported file type: {ext}")


def _chunk_text(text: str, max_chars: int = 3200) -> Iterable[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_length = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_length + len(para) + 2 > max_chars and current:
            yield "\n\n".join(current)
            current = [para]
            current_length = len(para)
        else:
            current.append(para)
            current_length += len(para) + 2
    if current:
        yield "\n\n".join(current)


def _call_model(manager, prompt: str) -> str:
    state = getattr(manager, "_history_enabled", True)
    manager.set_history_enabled(False)
    try:
        return manager.generate(prompt)
    finally:
        manager.set_history_enabled(state)


def summarize_document(manager, path: Path) -> str:
    if manager is None or not manager.is_loaded() or manager.is_image_backend():
        raise RuntimeError("A text-capable model must be loaded before analyzing a document.")
    text = extract_text(path)
    chunks = list(_chunk_text(text))
    if not chunks:
        raise DocumentReadError("Document did not contain any text to analyze.")
    summaries: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            f"You are a helpful assistant. Provide a concise summary of section {idx} of {path.name}.\n\n"
            f"Section:\n{chunk}\n\nSummary:"
        )
        summaries.append(_call_model(manager, prompt))
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        combined = "\n\n".join(f"Section {i+1}: {s}" for i, s in enumerate(summaries))
        merge_prompt = (
            f"Combine the following section summaries of {path.name} into a cohesive overall summary. "
            f"Limit the response to around 200 words.\n\n{combined}\n\nOverall Summary:"
        )
        final_summary = _call_model(manager, merge_prompt)
    return f"Summary of {path.name}:\n{final_summary.strip()}"
