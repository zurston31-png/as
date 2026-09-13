"""Provider registry and format sniffing."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..models import Match
from . import henrik, riot_official
from .base import Provider, ProviderError
from .files import FileProvider
from .henrik import HenrikProvider
from .local_client import LocalClientProvider
from .riot_official import RiotOfficialProvider

PROVIDER_NAMES = ("henrik", "riot", "local", "file")


def detect_format(payload: Any) -> str:
    """Identify a payload without knowing where it came from."""
    if riot_official.is_match_payload(payload):
        return riot_official.PAYLOAD_FORMAT
    if henrik.is_match_payload(payload):
        return henrik.PAYLOAD_FORMAT
    data = henrik.unwrap(payload)
    if riot_official.is_match_payload(data):
        return riot_official.PAYLOAD_FORMAT
    return ""


def parse_payload(
    payload: Any,
    fmt: str = "",
    assets: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[Match]:
    fmt = fmt or detect_format(payload)
    if fmt == riot_official.PAYLOAD_FORMAT:
        data = payload if riot_official.is_match_payload(payload) else henrik.unwrap(payload)
        return riot_official.parse_match(data, assets)
    if fmt == henrik.PAYLOAD_FORMAT:
        return henrik.parse_match(payload)
    return None


def iter_payloads(payload: Any) -> Iterator[Any]:
    """Yield every match payload inside a document (single match or list)."""
    fmt = detect_format(payload)
    if fmt == riot_official.PAYLOAD_FORMAT:
        data = payload if riot_official.is_match_payload(payload) else henrik.unwrap(payload)
        if isinstance(data, list):
            for item in data:
                if riot_official.is_match_payload(item):
                    yield item
        elif riot_official.is_match_payload(data):
            yield data
        return
    yield from henrik.iter_match_payloads(payload)


def build_provider(config: Any) -> Provider:
    """Instantiate the provider named in the config."""
    name = (getattr(config, "provider", "") or "henrik").lower()
    if name == "henrik":
        return HenrikProvider(
            api_key=config.henrik_api_key, region=config.region,
            platform=config.platform,
        )
    if name == "riot":
        return RiotOfficialProvider(
            api_key=config.riot_api_key, region=config.region,
            assets=config.load_assets(),
        )
    if name == "local":
        return LocalClientProvider(
            region=config.region, shard=config.shard,
            assets=config.load_assets(),
        )
    if name == "file":
        return FileProvider(paths=list(config.file_paths))
    raise ProviderError(
        f"unknown provider {name!r} (expected one of {', '.join(PROVIDER_NAMES)})"
    )


__all__ = [
    "FileProvider", "HenrikProvider", "LocalClientProvider", "Provider",
    "ProviderError", "RiotOfficialProvider", "build_provider", "detect_format",
    "iter_payloads", "parse_payload", "PROVIDER_NAMES",
]
