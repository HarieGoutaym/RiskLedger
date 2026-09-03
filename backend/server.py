"""
server.py — FastAPI Server Entrypoint shim for RiskLedger.
Exports `app` from `backend.app.main`.
"""

from backend.app.main import app

__all__ = ["app"]
