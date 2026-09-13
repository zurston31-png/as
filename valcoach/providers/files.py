"""Read matches from JSON files on disk.

Useful for three things: the bundled demo, importing a payload someone dumped
from another tool, and keeping the test suite entirely offline.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterator, List, Optional

from ..models import Match


class FileProvider:
    name = "file"
    payload_format = ""      # sniffed per payload

    def __init__(self, paths: Optional[List[str]] = None):
        self.paths = paths or []

    def iter_payloads(self) -> Iterator[Any]:
        for pattern in self.paths:
            expanded = sorted(glob.glob(os.path.expanduser(pattern))) or [pattern]
            for path in expanded:
                if os.path.isdir(path):
                    expanded_dir = sorted(glob.glob(os.path.join(path, "*.json")))
                    for nested in expanded_dir:
                        yield from self._load(nested)
                    continue
                yield from self._load(path)

    @staticmethod
    def _load(path: str) -> Iterator[Any]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data

    def parse(self, payload: Any) -> Optional[Match]:
        from . import parse_payload

        return parse_payload(payload)
