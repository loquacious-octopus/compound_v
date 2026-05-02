"""HTTP diagnostics for container startup failures.

This server is used only during infrastructure validation. It starts before the
miner API so a remote pod can expose startup logs even when model loading fails
before port 10006 would normally become healthy.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse


LOG_ROOT = Path("/tmp/404-startup-logs")
app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "diagnostic"}


@app.get("/status")
def status() -> dict:
    return {"status": "diagnostic", "payload": {"message": "startup diagnostics mode"}}


@app.get("/logs")
def list_logs() -> dict[str, list[str]]:
    if not LOG_ROOT.exists():
        return {"logs": []}
    return {"logs": sorted(path.name for path in LOG_ROOT.glob("*.log"))}


@app.get("/logs/{name}", response_class=PlainTextResponse)
def get_log(name: str) -> str:
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="invalid log name")
    path = LOG_ROOT / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="log not found")
    return path.read_text(encoding="utf-8", errors="replace")[-200_000:]


def main() -> None:
    import os

    import uvicorn

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("STARTUP_DIAGNOSTICS_PORT", "10006")),
        log_level=os.environ.get("STARTUP_DIAGNOSTICS_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
