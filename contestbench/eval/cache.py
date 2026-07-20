"""Disk cache for model calls, keyed by (model, params, prompt).

Every call is cached so reruns, added models, or a mid-run crash never re-pay
for completed calls (spec 5, reproducibility).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DiskCache:
    def __init__(self, directory) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, params: dict, prompt: str) -> str:
        payload = json.dumps(
            {"model": model, "params": params, "prompt": prompt},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, model: str, params: dict, prompt: str) -> str | None:
        path = self._path(self._key(model, params, prompt))
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def put(self, model: str, params: dict, prompt: str, response: str) -> None:
        key = self._key(model, params, prompt)
        self._path(key).write_text(
            json.dumps({"model": model, "params": params,
                        "prompt": prompt, "response": response}),
            encoding="utf-8",
        )
