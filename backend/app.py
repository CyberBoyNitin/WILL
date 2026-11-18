"""ASGI entry point: ``uvicorn backend.app:app``."""
from backend.server.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
