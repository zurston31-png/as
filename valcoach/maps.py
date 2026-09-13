"""Map knowledge: callout names and death-spot clustering.

Riot's match data gives death positions as raw game coordinates. Callout names
(``A Main``, ``Heaven``, ...) are not in match data, but the public
``valorant-api.com`` map endpoint publishes each map's callout list with
coordinates in the same space — so ``valcoach assets`` downloads them once and
the nearest callout to a death position becomes its name.

Without that download nothing breaks: positions are still clustered, and a
cluster is reported by coordinates plus how tightly packed it is.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import assets_path

# Deaths within this many game units are treated as "the same spot".
# ~1000 units is roughly a small room.
CLUSTER_RADIUS = 1100.0
# Beyond this the nearest callout is too far away to be worth naming.
CALLOUT_MAX_DISTANCE = 2600.0

AGENT_ROLES_PATH = os.path.join(os.path.dirname(__file__), "data", "agent_roles.json")


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def agent_role(agent: str) -> str:
    roles = _load_json(AGENT_ROLES_PATH) or {}
    return roles.get((agent or "").strip().lower(), "unknown")


@dataclass
class Callout:
    region: str
    super_region: str
    x: float
    y: float

    @property
    def label(self) -> str:
        if self.super_region and self.region:
            return f"{self.super_region} {self.region}"
        return self.region or self.super_region or ""


class MapIndex:
    """Callout lookup, loaded from the refreshable asset file."""

    def __init__(self, assets: Optional[Dict[str, Any]] = None):
        self.assets = assets if assets is not None else (_load_json(assets_path()) or {})
        self._callouts: Dict[str, List[Callout]] = {}
        for name, entry in (self.assets.get("maps") or {}).items():
            items = []
            for c in (entry or {}).get("callouts") or []:
                try:
                    items.append(
                        Callout(
                            region=str(c.get("region") or c.get("regionName") or ""),
                            super_region=str(
                                c.get("super_region") or c.get("superRegionName") or ""
                            ),
                            x=float((c.get("location") or c).get("x")),
                            y=float((c.get("location") or c).get("y")),
                        )
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
            if items:
                self._callouts[name.strip().lower()] = items

    @property
    def has_callouts(self) -> bool:
        return bool(self._callouts)

    def callout(self, map_name: str, x: Optional[float], y: Optional[float]) -> str:
        if x is None or y is None:
            return ""
        items = self._callouts.get((map_name or "").strip().lower())
        if not items:
            return ""
        best, best_dist = "", float("inf")
        for c in items:
            dist = math.hypot(c.x - x, c.y - y)
            if dist < best_dist:
                best, best_dist = c.label, dist
        return best if best_dist <= CALLOUT_MAX_DISTANCE else ""

    def describe(self, map_name: str, x: Optional[float], y: Optional[float]) -> str:
        name = self.callout(map_name, x, y)
        if name:
            return name
        if x is None or y is None:
            return "unknown position"
        return f"({int(x)}, {int(y)})"


@dataclass
class Cluster:
    x: float
    y: float
    members: List[int]          # indices into the input sequence
    radius: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)


def cluster_points(
    points: Sequence[Tuple[Optional[float], Optional[float]]],
    radius: float = CLUSTER_RADIUS,
) -> List[Cluster]:
    """Greedy single-pass spatial clustering.

    Deliberately simple: with tens to hundreds of deaths per report, a leader
    algorithm finds "you keep dying here" just as well as k-means and needs no
    tuning or dependencies.
    """
    clusters: List[Cluster] = []
    for i, (x, y) in enumerate(points):
        if x is None or y is None:
            continue
        placed = False
        for c in clusters:
            if math.hypot(c.x - x, c.y - y) <= radius:
                n = c.size
                c.x = (c.x * n + x) / (n + 1)
                c.y = (c.y * n + y) / (n + 1)
                c.members.append(i)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(x=float(x), y=float(y), members=[i]))

    for c in clusters:
        spread = 0.0
        for i in c.members:
            x, y = points[i]
            if x is None or y is None:
                continue
            spread = max(spread, math.hypot(c.x - float(x), c.y - float(y)))
        c.radius = spread
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def refresh_assets(
    out_path: Optional[str] = None,
    base_url: str = "https://valorant-api.com/v1",
    timeout: float = 30.0,
) -> Dict[str, int]:
    """Download map callouts and UUID→name tables from valorant-api.com."""
    from .webreq import get_json

    assets: Dict[str, Any] = {"maps": {}, "agents": {}, "weapons": {}}

    maps = (get_json(f"{base_url}/maps", timeout=timeout) or {}).get("data") or []
    for entry in maps:
        name = str(entry.get("displayName") or "").strip()
        if not name:
            continue
        assets["maps"][name] = {
            "uuid": entry.get("uuid"),
            "x_multiplier": entry.get("xMultiplier"),
            "y_multiplier": entry.get("yMultiplier"),
            "x_scalar_to_add": entry.get("xScalarToAdd"),
            "y_scalar_to_add": entry.get("yScalarToAdd"),
            "callouts": [
                {
                    "region": c.get("regionName"),
                    "super_region": c.get("superRegionName"),
                    "location": c.get("location"),
                }
                for c in (entry.get("callouts") or [])
            ],
        }

    agents = (
        get_json(f"{base_url}/agents", params={"isPlayableCharacter": "true"},
                 timeout=timeout)
        or {}
    ).get("data") or []
    for entry in agents:
        uuid = str(entry.get("uuid") or "").lower()
        if uuid:
            assets["agents"][uuid] = str(entry.get("displayName") or "")

    weapons = (get_json(f"{base_url}/weapons", timeout=timeout) or {}).get("data") or []
    for entry in weapons:
        uuid = str(entry.get("uuid") or "").lower()
        if uuid:
            assets["weapons"][uuid] = str(entry.get("displayName") or "")

    target = out_path or assets_path()
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(assets, handle, separators=(",", ":"), sort_keys=True)
    return {
        "maps": len(assets["maps"]),
        "callouts": sum(len(m["callouts"]) for m in assets["maps"].values()),
        "agents": len(assets["agents"]),
        "weapons": len(assets["weapons"]),
    }
