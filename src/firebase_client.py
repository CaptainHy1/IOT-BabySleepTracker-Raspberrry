# src/firebase_client.py
import os, threading
from typing import Dict, Any
import firebase_admin
from firebase_admin import credentials, db

_init_lock = threading.Lock()
_initialized = False

def _ensure_init():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        cred_path = os.environ.get("FIREBASE_CRED_JSON")
        db_url   = os.environ.get("FIREBASE_DB_URL")
        if not cred_path or not db_url:
            raise RuntimeError("Missing FIREBASE_CRED_JSON or FIREBASE_DB_URL env vars.")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        _initialized = True

def push_realtime(payload: Dict[str, Any], base_path: str | None = None):
    import os
    if os.environ.get("FIREBASE_SKIP") in ("1","true","yes","on"):
        return  # skip silently
    _ensure_init()
    base = base_path or os.environ.get("FIREBASE_BASE_PATH") or "sleepData"
    db.reference(base).push(payload)
