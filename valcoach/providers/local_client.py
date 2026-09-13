"""Local Valorant client provider — full match data, no API key.

While VALORANT is running it exposes a loopback API, authenticated by a
``lockfile`` that the Riot Client writes with a per-session password. With the
token pair from that API we can call Riot's own player-data endpoints
(``pd.<shard>.a.pvp.net``) as the logged-in player and get match details in the
official format, at full fidelity, with no third-party service in the middle.

Requirements: runs on the machine playing the game (Windows, or Wine/Proton
layouts if the paths resolve), with the client open. The loopback endpoint uses
a self-signed certificate, so verification is disabled **for 127.0.0.1 only**.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import Match
from ..webreq import get_json
from .base import ProviderError
from .riot_official import PAYLOAD_FORMAT, parse_match

# Sent as X-Riot-ClientPlatform; the pd endpoints reject requests without it.
CLIENT_PLATFORM = base64.b64encode(
    json.dumps(
        {
            "platformType": "PC",
            "platformOS": "Windows",
            "platformOSVersion": "10.0.19042.1.256.64bit",
            "platformChipset": "Unknown",
        },
        separators=(",", ":"),
    ).encode()
).decode()

SHARD_BY_REGION = {"na": "na", "latam": "na", "br": "na", "eu": "eu", "tr": "eu",
                   "ru": "eu", "ap": "ap", "kr": "kr", "jp": "ap"}


def lockfile_candidates() -> List[str]:
    paths: List[str] = []
    env = os.environ.get("VALCOACH_LOCKFILE")
    if env:
        paths.append(env)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        paths.append(os.path.join(local, "Riot Games", "Riot Client", "Config",
                                  "lockfile"))
    home = os.path.expanduser("~")
    paths.append(
        os.path.join(home, "AppData", "Local", "Riot Games", "Riot Client", "Config",
                     "lockfile")
    )
    # Common Proton/Wine prefix layout.
    paths.append(
        os.path.join(home, ".steam", "steam", "steamapps", "compatdata", "pfx",
                     "drive_c", "users", "steamuser", "AppData", "Local",
                     "Riot Games", "Riot Client", "Config", "lockfile")
    )
    return paths


def log_candidates() -> List[str]:
    paths: List[str] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        paths.append(os.path.join(local, "VALORANT", "Saved", "Logs",
                                  "ShooterGame.log"))
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, "AppData", "Local", "VALORANT", "Saved", "Logs",
                              "ShooterGame.log"))
    return paths


def read_lockfile() -> Dict[str, Any]:
    """Parse ``name:pid:port:password:protocol``."""
    for path in lockfile_candidates():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                parts = handle.read().strip().split(":")
        except OSError:
            continue
        if len(parts) >= 5:
            return {
                "name": parts[0], "pid": parts[1], "port": int(parts[2]),
                "password": parts[3], "protocol": parts[4], "path": path,
            }
    raise ProviderError(
        "VALORANT lockfile not found — start the game (or set VALCOACH_LOCKFILE)"
    )


def client_version_from_logs() -> str:
    """Read the running build string out of ShooterGame.log."""
    pattern = re.compile(r"CI server version:\s*([\w.\-]+)")
    for path in log_candidates():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    found = pattern.search(line)
                    if found:
                        return found.group(1).strip()
        except OSError:
            continue
    return ""


class LocalClientProvider:
    """Reads match history straight from the running game client."""

    name = "local"
    payload_format = PAYLOAD_FORMAT

    def __init__(
        self,
        region: str = "na",
        shard: str = "",
        client_version: str = "",
        timeout: float = 30.0,
        assets: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.region = (region or "na").lower()
        self.shard = (shard or SHARD_BY_REGION.get(self.region, self.region)).lower()
        self.timeout = timeout
        self.assets = assets or {}
        self._lock: Optional[Dict[str, Any]] = None
        self._tokens: Optional[Dict[str, str]] = None
        self._client_version = client_version

    # -- local auth
    @property
    def lock(self) -> Dict[str, Any]:
        if self._lock is None:
            self._lock = read_lockfile()
        return self._lock

    def _local_headers(self) -> Dict[str, str]:
        secret = base64.b64encode(f"riot:{self.lock['password']}".encode()).decode()
        return {"Authorization": f"Basic {secret}"}

    def local_get(self, path: str) -> Any:
        return get_json(
            f"https://127.0.0.1:{self.lock['port']}{path}",
            headers=self._local_headers(),
            timeout=self.timeout,
            retries=1,
            insecure=True,        # loopback only: the client's cert is self-signed
        )

    @property
    def tokens(self) -> Dict[str, str]:
        if self._tokens is None:
            data = self.local_get("/entitlements/v1/token") or {}
            access = data.get("accessToken")
            entitlement = data.get("token")
            subject = data.get("subject")
            if not access or not entitlement:
                raise ProviderError(
                    "client did not return session tokens — is a player signed in?"
                )
            self._tokens = {
                "access": str(access), "entitlement": str(entitlement),
                "puuid": str(subject or ""),
            }
        return self._tokens

    @property
    def client_version(self) -> str:
        if not self._client_version:
            self._client_version = client_version_from_logs()
        return self._client_version

    def _remote_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.tokens['access']}",
            "X-Riot-Entitlements-JWT": self.tokens["entitlement"],
            "X-Riot-ClientPlatform": CLIENT_PLATFORM,
        }
        if self.client_version:
            headers["X-Riot-ClientVersion"] = self.client_version
        return headers

    def _pd(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return get_json(
            f"https://pd.{self.shard}.a.pvp.net{path}",
            params=params, headers=self._remote_headers(), timeout=self.timeout,
        )

    # -- Provider protocol
    def resolve_player(self, riot_id: str = "") -> Tuple[str, str]:
        """The signed-in player is the only identity the client can speak for."""
        puuid = self.tokens["puuid"]
        if not puuid:
            raise ProviderError("client session has no puuid")
        name = ""
        try:
            data = self._pd("/name-service/v2/players", params=None)
            if isinstance(data, list) and data:
                entry = data[0]
                name = f"{entry.get('GameName', '')}#{entry.get('TagLine', '')}"
        except Exception:  # noqa: BLE001 - identity is cosmetic here
            pass
        return puuid, name or riot_id

    def current_match_id(self) -> Optional[str]:
        """Match id if the player is in a live game (used by ``valcoach watch``)."""
        puuid = self.tokens["puuid"]
        for host, path in (
            (f"https://glz-{self.region}-1.{self.shard}.a.pvp.net",
             f"/core-game/v1/players/{puuid}"),
            (f"https://glz-{self.region}-1.{self.shard}.a.pvp.net",
             f"/pregame/v1/players/{puuid}"),
        ):
            try:
                data = get_json(host + path, headers=self._remote_headers(),
                                timeout=self.timeout, retries=1)
            except Exception:  # noqa: BLE001 - 404 simply means "not in a game"
                continue
            if isinstance(data, dict) and data.get("MatchID"):
                return str(data["MatchID"])
        return None

    def recent_match_ids(self, puuid: str = "", count: int = 10,
                         queue: Optional[str] = None) -> List[str]:
        puuid = puuid or self.tokens["puuid"]
        data = self._pd(
            f"/match-history/v1/history/{puuid}",
            params={"startIndex": 0, "endIndex": max(1, int(count)),
                    "queue": queue or None},
        )
        history = (data or {}).get("History") or (data or {}).get("history") or []
        out = []
        for entry in history:
            mid = entry.get("MatchID") or entry.get("matchId")
            if mid:
                out.append(str(mid))
        return out

    def fetch_match(self, match_id: str) -> Any:
        return self._pd(f"/match-details/v1/matches/{match_id}")

    def parse(self, payload: Any) -> Optional[Match]:
        return parse_match(payload, self.assets)
