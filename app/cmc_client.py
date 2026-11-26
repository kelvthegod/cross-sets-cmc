"""
CoinMarketCap Pro API Async Client for Cross Sets (Data & Visualisation Track).
- Loads CMC_API_KEY strictly from environment (.env), never hardcoded in source.
- Resolves stable numeric CMC IDs via 0-credit /v1/cryptocurrency/map.
- Handles HTTP 429 / error_code 1007 with exponential backoff and TTL caching.
- Redacts API keys from all telemetry and /api/v1/evidence responses.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class CMCClientError(Exception):
    def __init__(self, message: str, status_code: int = 502, error_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class CMCAsyncClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_ttl: int = 60,
    ) -> None:
        self.api_key = (api_key or os.getenv("CMC_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("CMC_BASE_URL", "https://pro-api.coinmarketcap.com")).rstrip("/")
        self.cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", str(cache_ttl)))
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self.evidence_log: List[Dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_coinmarketcap_api_key_here")

    def _cache_key(self, path: str, params: Dict[str, Any]) -> str:
        sorted_items = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        return f"{path}?{sorted_items}"

    def _record_evidence(
        self,
        endpoint: str,
        params: Dict[str, Any],
        payload: Dict[str, Any],
        from_cache: bool = False,
    ) -> None:
        status = payload.get("status", {})
        sample_data = payload.get("data")
        if isinstance(sample_data, list):
            trimmed: Any = sample_data[:3]
        elif isinstance(sample_data, dict):
            trimmed = dict(list(sample_data.items())[:3])
        else:
            trimmed = sample_data

        entry = {
            "timestamp_utc": status.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint,
            "full_url": f"{self.base_url}{endpoint}",
            "headers": {"X-CMC_PRO_API_KEY": "REDACTED_***"},
            "params": {k: v for k, v in params.items() if v is not None},
            "credit_count": status.get("credit_count", 0),
            "error_code": status.get("error_code", 0),
            "cached": from_cache,
            "response_preview": {
                "status": status,
                "data_sample": trimmed,
            },
        }
        self.evidence_log.insert(0, entry)
        del self.evidence_log[30:]

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_override: Optional[int] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        key = self._cache_key(endpoint, clean_params)
        ttl = self.cache_ttl if ttl_override is None else ttl_override

        now = time.time()
        if key in self._cache:
            expires_at, cached_payload = self._cache[key]
            if now < expires_at:
                self._record_evidence(endpoint, clean_params, cached_payload, from_cache=True)
                return cached_payload

        if not self.is_configured:
            raise CMCClientError(
                "CMC_API_KEY is missing in environment (.env).",
                status_code=401,
            )

        headers = {
            "Accept": "application/json",
            "X-CMC_PRO_API_KEY": self.api_key,
        }

        backoff = 1.0
        async with httpx.AsyncClient(timeout=15.0) as client:
            for attempt in range(max_retries):
                resp = await client.get(f"{self.base_url}{endpoint}", params=clean_params, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                data = resp.json()
                status_obj = data.get("status", {})
                err_code = status_obj.get("error_code", 0)
                if resp.status_code >= 400 or (err_code not in (0, "0", None)):
                    if str(err_code) == "1007" and attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    raise CMCClientError(
                        status_obj.get("error_message", f"CMC HTTP {resp.status_code}"),
                        status_code=resp.status_code,
                        error_code=int(err_code) if str(err_code).isdigit() else None,
                    )

                self._cache[key] = (time.time() + ttl, data)
                self._record_evidence(endpoint, clean_params, data, from_cache=False)
                return data

        raise CMCClientError("CoinMarketCap API rate limit exceeded.", status_code=429, error_code=1007)
