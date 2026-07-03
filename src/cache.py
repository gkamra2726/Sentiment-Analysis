"""
cache.py
--------
Simple file-based cache for API responses (news + market data).
Avoids redundant downloads on repeated runs within the TTL window.
"""

import json
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import CACHE_DIR
from logger import get_logger

log = get_logger(__name__)


def _key_path(key: str, ext: str = "pkl") -> Path:
    safe = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{safe}.{ext}"


def cache_set(key: str, value: Any, ttl_hours: float = 6) -> None:
    """Persist value to disk with a timestamp."""
    path = _key_path(key)
    payload = {
        "ts":    datetime.now(tz=timezone.utc).isoformat(),
        "ttl_h": ttl_hours,
        "data":  value,
    }
    try:
        path.write_bytes(pickle.dumps(payload))
        log.debug("Cache SET  key=%s → %s", key[:50], path.name)
    except Exception as exc:
        log.warning("Cache write failed: %s", exc)


def cache_get(key: str) -> Optional[Any]:
    """Return cached value if not expired, else None."""
    path = _key_path(key)
    if not path.exists():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
        ts      = datetime.fromisoformat(payload["ts"])
        ttl_h   = payload.get("ttl_h", 6)
        age_h   = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600
        if age_h > ttl_h:
            log.debug("Cache STALE key=%s  age=%.1fh > ttl=%.1fh", key[:50], age_h, ttl_h)
            return None
        log.debug("Cache HIT   key=%s  age=%.1fh", key[:50], age_h)
        return payload["data"]
    except Exception as exc:
        log.warning("Cache read failed: %s", exc)
        return None


def cache_clear() -> int:
    """Delete all cached files. Returns count deleted."""
    n = 0
    for f in CACHE_DIR.glob("*.pkl"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    log.info("Cache cleared (%d files)", n)
    return n
