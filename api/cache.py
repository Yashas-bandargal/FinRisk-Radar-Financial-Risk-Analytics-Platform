"""
FinRisk Radar — Cache Layer
Redis cache with in-memory fallback.
Caches LLM responses, risk scores, and retrieval results.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)

# Try Redis; fall back to in-memory dict
try:
    import redis
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    _redis = redis.from_url(REDIS_URL, decode_responses=True)
    _redis.ping()
    USE_REDIS = True
    log.info(f"Redis connected: {REDIS_URL}")
except Exception as e:
    log.info(f"Redis unavailable ({e}), using in-memory cache")
    USE_REDIS = False
    _redis = None

_mem_cache: dict = {}
DEFAULT_TTL = 3600  # 1 hour


def _make_key(prefix: str, **kwargs) -> str:
    raw = json.dumps({"prefix": prefix, **kwargs}, sort_keys=True)
    return f"finrisk:{hashlib.md5(raw.encode()).hexdigest()}"


def get(key: str) -> Optional[Any]:
    if USE_REDIS:
        try:
            val = _redis.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            log.warning(f"Redis get error: {e}")
    else:
        entry = _mem_cache.get(key)
        if entry:
            if datetime.now() < entry["expires"]:
                return entry["data"]
            else:
                del _mem_cache[key]
    return None


def set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
    try:
        if USE_REDIS:
            _redis.setex(key, ttl, json.dumps(value, default=str))
        else:
            _mem_cache[key] = {
                "data":    value,
                "expires": datetime.now() + timedelta(seconds=ttl),
            }
        return True
    except Exception as e:
        log.warning(f"Cache set error: {e}")
        return False


def delete(key: str) -> bool:
    try:
        if USE_REDIS:
            _redis.delete(key)
        else:
            _mem_cache.pop(key, None)
        return True
    except Exception:
        return False


def cached(prefix: str, ttl: int = DEFAULT_TTL):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = _make_key(prefix, args=str(args), kwargs=str(kwargs))
            result = get(key)
            if result is not None:
                log.debug(f"Cache hit: {prefix}")
                return result
            result = func(*args, **kwargs)
            set(key, result, ttl)
            return result
        return wrapper
    return decorator


def get_stats() -> dict:
    if USE_REDIS:
        info = _redis.info("stats")
        return {
            "backend":    "redis",
            "hits":       info.get("keyspace_hits", 0),
            "misses":     info.get("keyspace_misses", 0),
            "keys":       _redis.dbsize(),
        }
    return {
        "backend": "memory",
        "keys":    len(_mem_cache),
        "active":  sum(1 for v in _mem_cache.values() if datetime.now() < v["expires"]),
    }
