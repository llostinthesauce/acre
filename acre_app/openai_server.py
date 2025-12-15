from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Tuple
from urllib.parse import urlsplit


def _now_ts() -> int:
    return int(time.time())


def _chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _as_bool(value: Any) -> bool:
    return bool(value) is True


def _parse_stop(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item:
                out.append(item)
        return tuple(out)
    return None


class OpenAIHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        manager_getter: Callable[[], Any],
        auth_token: Optional[str] = None,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.manager_getter = manager_getter
        self.auth_token = (auth_token or "").strip() or None
        self.request_log: list[dict[str, Any]] = []
        self.request_log_lock = threading.Lock()
        self.request_log_max = 100


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ACREOpenAI/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Silence default request logs (privacy-first).
        return

    def _append_log(self, status: int, *, stream: Optional[bool] = None) -> None:
        try:
            server = self.server
            lock = getattr(server, "request_log_lock", None)
            log = getattr(server, "request_log", None)
            limit = int(getattr(server, "request_log_max", 100))
            if lock is None or log is None:
                return
            entry: dict[str, Any] = {
                "ts": _now_ts(),
                "method": str(getattr(self, "command", "") or ""),
                "path": urlsplit(self.path).path,
                "status": int(status),
            }
            if stream is not None:
                entry["stream"] = bool(stream)
            with lock:
                log.append(entry)
                if limit > 0 and len(log) > limit:
                    del log[: max(0, len(log) - limit)]
        except Exception:
            return

    def _send_json(self, status: int, payload: Any, *, extra_headers: Optional[dict[str, str]] = None) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        self._append_log(status, stream=False)

    def _send_error_json(
        self,
        status: int,
        message: str,
        *,
        code: str = "invalid_request_error",
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        payload = {"error": {"message": message, "type": error_type, "code": code}}
        if param:
            payload["error"]["param"] = param
        self._send_json(status, payload, extra_headers=extra_headers)

    def _require_auth(self) -> bool:
        token = getattr(self.server, "auth_token", None)
        if not token:
            return True
        header = str(self.headers.get("Authorization") or "")
        expected = f"Bearer {token}"
        if header.strip() == expected:
            return True
        self._send_error_json(
            401,
            "Unauthorized.",
            code="unauthorized",
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_json_body(self) -> Optional[dict]:
        raw_len = self.headers.get("Content-Length")
        if not raw_len:
            return {}
        try:
            length = int(raw_len)
        except Exception:
            self._send_error_json(400, "Invalid Content-Length.", code="bad_request")
            return None
        if length < 0:
            self._send_error_json(400, "Invalid Content-Length.", code="bad_request")
            return None
        if length > 10_000_000:
            self._send_error_json(413, "Request too large.", code="request_too_large")
            return None
        try:
            body = self.rfile.read(length)
        except Exception:
            self._send_error_json(400, "Failed to read request body.", code="bad_request")
            return None
        if not body:
            return {}
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_error_json(400, "Request body must be valid JSON.", code="bad_request")
            return None
        if not isinstance(data, dict):
            self._send_error_json(400, "Request body must be a JSON object.", code="bad_request")
            return None
        return data

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/models":
            self._handle_models()
            return
        self._send_error_json(404, "Not found.", code="not_found")

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        self._send_error_json(404, "Not found.", code="not_found")

    def _handle_models(self) -> None:
        mgr = getattr(self.server, "manager_getter", lambda: None)()
        created = _now_ts()
        items: list[dict] = []
        current = getattr(mgr, "current_model_name", None) if mgr else None
        names: list[str] = []
        if mgr and getattr(mgr, "list_models", None):
            try:
                names = list(mgr.list_models())
            except Exception:
                names = []
        if current and current not in names:
            names.insert(0, current)
        for name in names:
            items.append(
                {
                    "id": str(name),
                    "object": "model",
                    "created": created,
                    "owned_by": "acre",
                }
            )
        self._send_json(200, {"object": "list", "data": items})

    def _handle_chat_completions(self) -> None:
        data = self._read_json_body()
        if data is None:
            return

        mgr = getattr(self.server, "manager_getter", lambda: None)()
        if not mgr or not getattr(mgr, "is_loaded", lambda: False)():
            self._send_error_json(400, "No model loaded. Load a model in the ACRE GUI first.", param="model")
            return

        current_model = getattr(mgr, "current_model_name", None) or "unknown"
        requested_model = data.get("model")
        if requested_model and str(requested_model) != str(current_model):
            self._send_error_json(
                400,
                f"Requested model '{requested_model}' is not loaded. Load '{requested_model}' in the ACRE GUI, or omit 'model' to use '{current_model}'.",
                param="model",
            )
            return

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error_json(400, "Request must include a non-empty 'messages' array.", param="messages")
            return

        stream = _as_bool(data.get("stream"))
        max_tokens = data.get("max_tokens")
        temperature = data.get("temperature")
        stop = _parse_stop(data.get("stop"))

        if stream:
            self._stream_chat_completion(mgr, current_model, messages, max_tokens, temperature, stop)
            return

        try:
            self._chat_completion(mgr, current_model, messages, max_tokens, temperature, stop)
        except RuntimeError as exc:
            message = str(exc)
            if "already in progress" in message.lower():
                self._send_error_json(409, message, code="model_busy")
                return
            self._send_error_json(500, message, code="server_error")
        except Exception as exc:
            self._send_error_json(500, f"Server error: {exc}", code="server_error")

    def _chat_completion(
        self,
        mgr: Any,
        model: str,
        messages: list,
        max_tokens: Any,
        temperature: Any,
        stop: Optional[Tuple[str, ...]],
    ) -> None:
        text = mgr.generate_messages(
            messages,
            max_tokens=max_tokens if max_tokens is not None else None,
            temperature=temperature if temperature is not None else None,
            stop=stop,
        )
        payload = {
            "id": _chat_id(),
            "object": "chat.completion",
            "created": _now_ts(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(200, payload)

    def _stream_chat_completion(
        self,
        mgr: Any,
        model: str,
        messages: list,
        max_tokens: Any,
        temperature: Any,
        stop: Optional[Tuple[str, ...]],
    ) -> None:
        cancel = threading.Event()
        request_id = _chat_id()
        created = _now_ts()
        had_error = False

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._append_log(200, stream=True)

        def send_event(obj: Any) -> bool:
            try:
                payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
                data = f"data: {payload}\n\n".encode("utf-8")
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False
            except Exception:
                return False

        # Initial chunk: include assistant role.
        if not send_event(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        ):
            return

        try:
            try:
                iterator = mgr.generate_stream_messages(
                    messages,
                    cancel_event=cancel,
                    max_tokens=max_tokens if max_tokens is not None else None,
                    temperature=temperature if temperature is not None else None,
                    stop=stop,
                )
                for chunk in iterator:
                    if cancel.is_set():
                        break
                    if not chunk:
                        continue
                    ok = send_event(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": str(chunk)}, "finish_reason": None}],
                        }
                    )
                    if not ok:
                        cancel.set()
                        try:
                            mgr.cancel_generation()
                        except Exception:
                            pass
                        break
            except RuntimeError as exc:
                had_error = True
                message = str(exc)
                code = "model_busy" if "already in progress" in message.lower() else "server_error"
                send_event({"error": {"message": message, "type": "server_error", "code": code}})
            except Exception as exc:
                had_error = True
                send_event({"error": {"message": f"Server error: {exc}", "type": "server_error", "code": "server_error"}})
        finally:
            # Final chunk (when successful), then [DONE].
            if not had_error:
                send_event(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
            send_event("[DONE]")


def start_openai_server(
    *,
    host: str = "127.0.0.1",
    port: int = 4891,
    manager_getter: Callable[[], Any],
    auth_token: Optional[str] = None,
) -> tuple[OpenAIHTTPServer, threading.Thread]:
    server = OpenAIHTTPServer(
        (host, int(port)),
        OpenAIRequestHandler,
        manager_getter=manager_getter,
        auth_token=auth_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_openai_server(server: Optional[OpenAIHTTPServer]) -> None:
    if not server:
        return
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()
    except Exception:
        pass
