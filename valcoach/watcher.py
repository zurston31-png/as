"""The always-on part: watch for new matches and review them as they land.

``valcoach watch`` polls your match history on an interval. When a new match
appears it is stored, analysed, and summarised — optionally with a written
review and a desktop notification. If the local game client is the provider, it
also notices when you are in a live match and checks again as soon as it ends,
instead of waiting out the interval.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .analysis.report import Report, build_report
from .config import Config
from .maps import MapIndex
from .providers import ProviderError, build_provider, iter_payloads, parse_payload
from .store import Store


@dataclass
class SyncResult:
    fetched: int = 0
    new_match_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def new_count(self) -> int:
        return len(self.new_match_ids)


def sync_matches(
    config: Config,
    store: Store,
    count: int = 10,
    puuid: str = "",
    log: Optional[Callable[[str], None]] = None,
) -> SyncResult:
    """Pull recent matches from the configured provider into the store."""
    say = log or (lambda _msg: None)
    result = SyncResult()
    provider = build_provider(config)
    fmt = getattr(provider, "payload_format", "") or ""

    # The file provider has no notion of "recent"; it ingests whatever it is given.
    if getattr(provider, "name", "") == "file":
        for payload in provider.iter_payloads():
            for single in iter_payloads(payload):
                match = parse_payload(single)
                if not match or not match.match_id:
                    continue
                result.fetched += 1
                if store.save_match(match, single, match.provider):
                    result.new_match_ids.append(match.match_id)
        return result

    target = puuid or config.puuid
    if not target:
        if not config.riot_id:
            raise ProviderError(
                "no player configured — run `valcoach init --riot-id Name#TAG`"
            )
        target, canonical = provider.resolve_player(config.riot_id)
        config.puuid = target
        if canonical:
            config.riot_id = canonical
        config.save()
        say(f"resolved {config.riot_id} → {target[:8]}…")

    payloads: List[Any] = []
    recent = getattr(provider, "recent_matches", None)
    if callable(recent):
        # HenrikDev returns whole matches in the history call: one round trip.
        payloads = list(recent(target, count, config.queue or None))
    else:
        for match_id in provider.recent_match_ids(target, count, config.queue or None):
            if store.has_match(match_id):
                continue
            try:
                payloads.append(provider.fetch_match(match_id))
            except Exception as exc:  # noqa: BLE001 - keep going through the list
                result.errors.append(f"{match_id}: {exc}")

    for payload in payloads:
        for single in iter_payloads(payload):
            match = parse_payload(single, fmt)
            if not match or not match.match_id:
                continue
            result.fetched += 1
            if store.has_match(match.match_id):
                continue
            if store.save_match(match, single, fmt or match.provider):
                result.new_match_ids.append(match.match_id)
                say(
                    f"new match {match.match_id[:8]}… {match.map_name} "
                    f"{match.queue}"
                )
    store.set_meta("last_sync", str(int(time.time())))
    return result


def notify(title: str, message: str) -> bool:
    """Best-effort desktop notification; silently does nothing if unavailable."""
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            script = (
                f'display notification {message!r} with title {title!r}'
            )
            subprocess.run(["osascript", "-e", script], check=False,
                           capture_output=True, timeout=5)
            return True
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            subprocess.run(["notify-send", title, message], check=False,
                           capture_output=True, timeout=5)
            return True
        if os.name == "nt" and shutil.which("powershell"):
            ps = (
                "[reflection.assembly]::loadwithpartialname('System.Windows.Forms');"
                "$n=New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon=[System.Drawing.SystemIcons]::Information;"
                f"$n.BalloonTipTitle={title!r};$n.BalloonTipText={message!r};"
                "$n.Visible=$true;$n.ShowBalloonTip(8000)"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           check=False, capture_output=True, timeout=10)
            return True
    except Exception:  # noqa: BLE001 - notifications are never load-bearing
        return False
    return False


def review_window(
    config: Config,
    store: Store,
    last: int = 10,
    puuid: str = "",
    strict: bool = False,
    map_index: Optional[MapIndex] = None,
) -> Report:
    """Build a report over the most recent `last` stored matches."""
    target = puuid or config.puuid or (store.resolve_puuid(config.riot_id) or "")
    if not target:
        raise ProviderError(
            "no player to analyse — run `valcoach init --riot-id Name#TAG` or pass --player"
        )
    matches = store.load_matches(
        puuid=target, limit=last, queue=config.queue or None
    )
    riot_id = config.riot_id
    if not riot_id and matches:
        player = matches[0].player(target)
        riot_id = player.ref.riot_id if player else ""
    return build_report(
        matches,
        puuid=target,
        riot_id=riot_id,
        trade_window_ms=config.trade_window_ms,
        map_index=map_index if map_index is not None else MapIndex(config.load_assets()),
        strict=strict,
        previous_reports=store.recent_reports(target, limit=3),
        queue_filter=config.queue,
    )


class Watcher:
    """Poll loop. Stops on Ctrl-C, or after `max_cycles` (used by tests)."""

    def __init__(
        self,
        config: Config,
        store: Store,
        interval: int = 180,
        count: int = 5,
        on_new_match: Optional[Callable[[Report, SyncResult], None]] = None,
        log: Optional[Callable[[str], None]] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.store = store
        self.interval = max(30, int(interval))
        self.count = count
        self.on_new_match = on_new_match
        self.log = log or (lambda msg: print(msg, flush=True))
        self.sleeper = sleeper
        self.cycles = 0

    def _live_match(self) -> Optional[str]:
        """If the provider can see a live game, report its id."""
        try:
            provider = build_provider(self.config)
        except Exception:  # noqa: BLE001
            return None
        getter = getattr(provider, "current_match_id", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 - not in a game, or client closed
            return None

    def cycle(self) -> SyncResult:
        self.cycles += 1
        try:
            result = sync_matches(
                self.config, self.store, count=self.count, log=self.log
            )
        except Exception as exc:  # noqa: BLE001 - a watcher must survive a bad poll
            self.log(f"sync failed: {exc}")
            return SyncResult(errors=[str(exc)])

        if result.new_count and self.on_new_match:
            report = review_window(
                self.config, self.store, last=max(1, self.count)
            )
            self.on_new_match(report, result)
        return result

    def run(self, max_cycles: Optional[int] = None) -> int:
        self.log(
            f"watching for new matches every {self.interval}s "
            f"(provider: {self.config.provider}) — Ctrl-C to stop"
        )
        try:
            while max_cycles is None or self.cycles < max_cycles:
                live = self._live_match()
                if live:
                    self.log(f"in a match ({live[:8]}…), waiting for it to finish")
                    self.sleeper(min(self.interval, 60))
                    self.cycles += 1
                    continue
                self.cycle()
                if max_cycles is not None and self.cycles >= max_cycles:
                    break
                self.sleeper(self.interval)
        except KeyboardInterrupt:
            self.log("\nstopped")
        return self.cycles
