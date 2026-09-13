"""Provider interface.

A provider is anything that can list recent match ids for a player and return
the raw payload for one match. Parsing lives next to the provider so the raw
payload can be re-parsed later (see ``Store.reindex``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple

from ..models import Match


class Provider(Protocol):
    name: str
    payload_format: str

    def resolve_player(self, riot_id: str) -> Tuple[str, str]:
        """Return (puuid, canonical 'Name#TAG')."""

    def recent_match_ids(self, puuid: str, count: int = 10, queue: Optional[str] = None) -> List[str]:
        ...

    def fetch_match(self, match_id: str) -> Any:
        """Return the raw provider payload for one match."""

    def parse(self, payload: Any) -> Optional[Match]:
        ...


class ProviderError(RuntimeError):
    pass
