from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .documents import DocumentReadError


WORD_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _keywords(text: str) -> set[str]:
    words = {w for w in WORD_RE.findall((text or "").lower()) if len(w) > 2}
    return {w for w in words if w not in STOPWORDS}


def _read_text_file(path: Path) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            data = path.read_text(encoding="latin-1", errors="replace")
        except Exception as exc:
            raise DocumentReadError(f"Unable to read file: {exc}") from exc
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = data.strip()
    if not cleaned:
        raise DocumentReadError("File appears to be empty.")
    return cleaned


def _chunk_text(text: str, *, max_chars: int = 2400) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0
    chunks: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def find_default_ocr_model(models_dir: Path) -> Optional[str]:
    try:
        items = sorted(models_dir.iterdir())
    except Exception:
        return None
    for item in items:
        if not item.is_dir():
            continue
        config = item / "config.json"
        if not config.exists():
            continue
        try:
            blob = config.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lowered = blob.lower()
        if '"model_type"' in lowered and "trocr" in lowered:
            return item.name
        name_lower = item.name.lower()
        if "trocr" in name_lower or ("ocr" in name_lower and "vision" not in name_lower):
            return item.name
    return None


def _pdf_page_text(path: Path) -> tuple[list[dict[str, str]], list[int]]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise DocumentReadError("PyMuPDF is required to read PDF files.") from exc
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise DocumentReadError(f"Unable to open PDF: {exc}") from exc
    chunks: list[dict[str, str]] = []
    ocr_pages: list[int] = []
    try:
        for idx, page in enumerate(doc, start=1):
            try:
                text = page.get_text() or ""
            except Exception:
                text = ""
            cleaned = str(text).strip()
            if cleaned:
                chunks.append({"source": f"p{idx}", "text": cleaned})
            else:
                ocr_pages.append(idx)
    finally:
        doc.close()
    return chunks, ocr_pages


def _pdf_pages_ocr(
    path: Path,
    pages: list[int],
    *,
    models_dir: Path,
    device_pref: str,
    cancel_event: Optional[threading.Event] = None,
) -> list[dict[str, str]]:
    model_name = find_default_ocr_model(models_dir)
    if not model_name:
        raise DocumentReadError(
            "This PDF appears to be scanned (no extractable text). OCR fallback requires an OCR model in your models folder (e.g., TrOCR)."
        )
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise DocumentReadError("PyMuPDF is required to read PDF files.") from exc

    from model_manager import ModelManager

    with tempfile.TemporaryDirectory(prefix="acre_ocr_") as tmp:
        tmp_dir = Path(tmp)
        ocr_mgr = ModelManager(models_dir=str(models_dir), history_dir=str(tmp_dir), device_pref=device_pref)
        ocr_mgr.set_history_enabled(False)
        ok, msg = ocr_mgr.load_model(model_name)
        if not ok:
            raise DocumentReadError(f"Failed to load OCR model '{model_name}': {msg}")
        try:
            doc = fitz.open(path)
        except Exception as exc:
            ocr_mgr.unload()
            raise DocumentReadError(f"Unable to open PDF: {exc}") from exc
        out: list[dict[str, str]] = []
        try:
            for page_num in pages:
                if cancel_event is not None and cancel_event.is_set():
                    raise DocumentReadError("Canceled.")
                try:
                    page = doc[page_num - 1]
                except Exception:
                    continue
                try:
                    pix = page.get_pixmap(dpi=160)
                except Exception:
                    try:
                        pix = page.get_pixmap()
                    except Exception:
                        continue
                img_path = tmp_dir / f"p{page_num}.png"
                try:
                    pix.save(str(img_path))
                except Exception:
                    continue
                try:
                    text = ocr_mgr.run_ocr(str(img_path))
                except Exception:
                    text = ""
                cleaned = str(text).strip()
                if cleaned:
                    out.append({"source": f"p{page_num}", "text": cleaned})
        finally:
            try:
                doc.close()
            except Exception:
                pass
            try:
                ocr_mgr.unload()
            except Exception:
                pass
        return out


def load_document_chunks(
    path: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    models_dir: Optional[Path] = None,
    device_pref: str = "auto",
) -> list[dict[str, str]]:
    if not path.exists():
        raise DocumentReadError(f"Document not found: {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        chunks, ocr_pages = _pdf_page_text(path)
        total_pages = len(chunks) + len(ocr_pages)
        total_chars = sum(len(str(item.get("text", ""))) for item in chunks)
        looks_scanned = (not chunks) or (total_pages >= 3 and total_chars < 200)

        # Run OCR only when the PDF appears to be scanned / mostly-image.
        if looks_scanned and ocr_pages:
            if models_dir is None:
                raise DocumentReadError("OCR fallback requires a models directory path.")
            chunks = _pdf_pages_ocr(
                path,
                ocr_pages,
                models_dir=models_dir,
                device_pref=device_pref,
                cancel_event=cancel_event,
            )
        if not chunks:
            raise DocumentReadError("No extractable text found in this PDF.")
        return chunks
    if ext in {".txt", ".md", ".markdown"}:
        text = _read_text_file(path)
        pieces = _chunk_text(text, max_chars=2400)
        return [{"source": f"s{idx}", "text": piece} for idx, piece in enumerate(pieces, start=1)]
    raise DocumentReadError(f"Unsupported file type: {ext}")


def select_relevant_excerpts(
    chunks: list[dict[str, str]],
    question: str,
    *,
    max_excerpts: int = 6,
    max_chars: int = 1400,
) -> list[dict[str, str]]:
    if not chunks:
        return []
    q_words = _keywords(question)

    scored: list[tuple[int, int, dict[str, str]]] = []
    for idx, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        score = len(q_words & _keywords(text)) if q_words else 0
        scored.append((score, idx, chunk))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    picked = [c for score, _, c in scored[: max(1, int(max_excerpts))]]

    out: list[dict[str, str]] = []
    for item in picked:
        src = str(item.get("source") or "").strip() or "doc"
        text = str(item.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 1)].rstrip() + "…"
        out.append({"source": src, "text": text})
    return out


def build_cited_prompt(doc_name: str, question: str, excerpts: list[dict[str, str]]) -> str:
    if not excerpts:
        raise DocumentReadError("No excerpts available to answer from.")
    lines: list[str] = []
    lines.append(
        "You are a helpful assistant. Answer the question using ONLY the provided document excerpts."
    )
    lines.append("Cite sources inline like [1], [2] matching the excerpt numbers.")
    lines.append("If the answer is not in the excerpts, say you don't know.")
    lines.append("")
    lines.append(f"Document: {doc_name}")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Excerpts:")
    for idx, ex in enumerate(excerpts, start=1):
        src = str(ex.get("source") or "").strip()
        text = str(ex.get("text") or "").strip()
        label = f" ({src})" if src else ""
        lines.append(f"[{idx}]{label} {text}")
        lines.append("")
    lines.append("Answer:")
    return "\n".join(lines).strip()
