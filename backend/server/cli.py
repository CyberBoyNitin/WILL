"""``will-serve`` console entry point."""
from __future__ import annotations

import argparse

import uvicorn

from backend.config import load_config


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Serve the WILL API")
    parser.add_argument("--host", default=cfg.server.host)
    parser.add_argument("--port", type=int, default=cfg.server.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=cfg.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
