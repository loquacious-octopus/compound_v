"""OpenAI-compatible router for local model servers.

The production Targon container has enough GPU memory to run the promoted
Qwen3 models one at a time across all 4 H200s. This router presents one stable
OpenAI-compatible endpoint to the generator and swaps the backing vLLM process
when a request targets a different local model.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response


app = FastAPI()
_lock = threading.Lock()
_active_model: str | None = None
_active_process: subprocess.Popen[str] | None = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).rstrip("/")


def _vision_model() -> str:
    return os.environ.get("CHUTES_VISION_MODEL") or os.environ.get("PRODUCTION_VISION_MODEL", "")


def _code_model() -> str:
    return os.environ.get("CHUTES_CODE_MODEL") or os.environ.get("PRODUCTION_CODE_MODEL", "")


def _target_base(model: str) -> str:
    if os.environ.get("LOCAL_LLM_ROUTER_MODE", "managed").strip().lower() == "managed":
        _ensure_model(model)
        return _env("LOCAL_MANAGED_BASE_URL", "http://127.0.0.1:8010/v1")
    if model == _code_model():
        return _env("LOCAL_CODE_BASE_URL", "http://127.0.0.1:8002/v1")
    if model == _vision_model():
        return _env("LOCAL_VISION_BASE_URL", "http://127.0.0.1:8001/v1")
    raise HTTPException(status_code=404, detail=f"No local model route for {model!r}")


def _model_kind(model: str) -> str:
    if model == _vision_model():
        return "vision"
    if model == _code_model():
        return "code"
    raise HTTPException(status_code=404, detail=f"No local model route for {model!r}")


def _wait_models(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = str(exc)
        if _active_process is not None and _active_process.poll() is not None:
            raise RuntimeError(f"vLLM exited while loading model; see {_log_path(_active_model or 'unknown')}")
        time.sleep(float(os.environ.get("LOCAL_LLM_HEALTH_INTERVAL_SECONDS", "10")))
    raise RuntimeError(f"Timed out waiting for local vLLM at {url}: {last_error}")


def _log_path(kind: str) -> Path:
    path = Path("/tmp/404-startup-logs")
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{kind}.log"


def _stop_active_model() -> None:
    global _active_model, _active_process
    if _active_process is None:
        _active_model = None
        return
    if _active_process.poll() is None:
        _active_process.terminate()
        try:
            _active_process.wait(timeout=float(os.environ.get("LOCAL_LLM_STOP_TIMEOUT_SECONDS", "60")))
        except subprocess.TimeoutExpired:
            _active_process.kill()
            _active_process.wait(timeout=30)
    _active_model = None
    _active_process = None


def _ensure_model(model: str) -> None:
    global _active_model, _active_process
    kind = _model_kind(model)
    with _lock:
        if _active_model == model and _active_process is not None and _active_process.poll() is None:
            return

        _stop_active_model()
        port = os.environ.get("LOCAL_MANAGED_PORT", "8010")
        tp = os.environ.get(f"LOCAL_{kind.upper()}_TENSOR_PARALLEL_SIZE", os.environ.get("LOCAL_MANAGED_TENSOR_PARALLEL_SIZE", "4"))
        gpus = os.environ.get(f"LOCAL_{kind.upper()}_CUDA_DEVICES", os.environ.get("LOCAL_MANAGED_CUDA_DEVICES", "0,1,2,3"))
        common_args = os.environ.get("LOCAL_VLLM_COMMON_ARGS", "--moe-backend triton")
        extra_args = os.environ.get(f"LOCAL_{kind.upper()}_VLLM_ARGS", os.environ.get("LOCAL_MANAGED_VLLM_ARGS", ""))
        timeout = float(os.environ.get("LOCAL_LLM_READY_TIMEOUT_SECONDS", "3600"))
        log_file = _log_path(kind)
        cmd = [
            "vllm",
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--served-model-name",
            model,
            "--tensor-parallel-size",
            tp,
            "--dtype",
            os.environ.get("LOCAL_VLLM_DTYPE", "auto"),
            "--trust-remote-code",
            "--generation-config",
            "vllm",
            *common_args.split(),
            *extra_args.split(),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[llm_router] starting {kind} model={model} gpus={gpus} tp={tp}\n")
            handle.write(f"[llm_router] command: {' '.join(cmd)}\n")
            handle.flush()
            _active_process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        _active_model = model
        try:
            _wait_models(f"http://127.0.0.1:{port}/v1/models", timeout)
        except Exception:
            _stop_active_model()
            raise


def _forward_json(url: str, body: dict[str, Any] | None = None, timeout: float = 300.0) -> tuple[int, bytes, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            content_type = response.headers.get("Content-Type", "application/json")
            return response.status, response.read(), content_type
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise HTTPException(status_code=502, detail=f"Local model server unavailable: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {
        "object": "list",
        "data": [
            {"id": _vision_model(), "object": "model"},
            {"id": _code_model(), "object": "model"},
        ],
    }


@app.post("/admin/preload")
async def preload(request: Request) -> dict[str, str]:
    payload = await request.json()
    model = payload.get("model")
    if model == "vision":
        model = _vision_model()
    elif model == "code":
        model = _code_model()
    if not isinstance(model, str) or not model:
        raise HTTPException(status_code=400, detail="model is required")
    _ensure_model(model)
    return {"status": "ready", "model": model}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise HTTPException(status_code=400, detail="model is required")

    timeout = float(os.environ.get("LOCAL_LLM_ROUTER_TIMEOUT", os.environ.get("CHUTES_TIMEOUT", "300")))
    status, body, content_type = _forward_json(f"{_target_base(model)}/chat/completions", payload, timeout=timeout)
    return Response(content=body, status_code=status, media_type=content_type)


@app.get("/v1/{path:path}")
def unsupported(path: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"Unsupported local LLM route: /v1/{path}"})


def main() -> None:
    import uvicorn

    host = os.environ.get("LOCAL_LLM_ROUTER_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL_LLM_ROUTER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOCAL_LLM_ROUTER_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
