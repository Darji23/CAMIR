"""FastAPI server and OpenAI-compatible proxy package for CAMIR."""

from camir.server.app import app, create_app

__all__ = ["app", "create_app"]
