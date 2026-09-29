"""Getting the signal in front of you.

Three independent channels, each best-effort: the terminal, a system
notification, and a sound. None of them is allowed to raise - a missing
`notify-send` should never stop a trade alert from reaching the dashboard.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from typing import Optional

from .config import NotifyConfig
from .models import Side, TradeSignal

log = logging.getLogger(__name__)

_RESET = "\033[0m"
_COLORS = {Side.LONG: "\033[1;32m", Side.SHORT: "\033[1;31m", Side.FLAT: "\033[1;33m"}


class Notifier:
    def __init__(self, config: NotifyConfig) -> None:
        self.cfg = config
        self.system = platform.system()

    def signal(self, signal: TradeSignal) -> None:
        if self.cfg.only_actionable and not signal.actionable:
            if self.cfg.console:
                self._console_quiet(signal)
            return
        if self.cfg.console:
            self._console(signal)
        title = f"{signal.side.value} {signal.symbol}" if signal.actionable else "No trade"
        body = signal.headline()
        if self.cfg.desktop:
            self.desktop(title, body)
        if self.cfg.sound:
            self.sound(signal.side)

    # ------------------------------------------------------------- console

    def _console(self, signal: TradeSignal) -> None:
        color = _COLORS.get(signal.side, "")
        bar = "=" * 62
        ai = signal.ai
        lines = [
            f"{color}{bar}",
            f"  {signal.side.value}  {signal.symbol}  {signal.timeframe}   "
            f"[{signal.ts.strftime('%H:%M:%S')}]",
            bar + _RESET,
            f"  Entry : {signal.entry:g}",
            f"  Stop  : {signal.stop:g}   ({signal.stop_distance:g} away)",
            f"  TP1   : {signal.tp1:g}   ({signal.rr1:g}R)",
            f"  TP2   : {signal.tp2:g}   ({signal.rr2:g}R)",
            f"  Size  : {signal.qty:g}   risking {signal.risk_pct * 100:.2f}% "
            f"(${signal.risk_amount:,.0f})",
        ]
        if ai and ai.rationale:
            lines.append(f"  AI    : {ai.decision.value} {ai.confidence:.2f} - {ai.rationale}")
        if signal.strategy_verdict:
            passed = [c.name for c in signal.strategy_verdict.checks if c.passed]
            lines.append(f"  Rules : {', '.join(passed)}")
        lines.append(color + bar + _RESET)
        print("\n".join(lines), flush=True)

    def _console_quiet(self, signal: TradeSignal) -> None:
        print(
            f"\033[2m[{signal.ts.strftime('%H:%M:%S')}] no trade - {signal.blocked_by}\033[0m",
            flush=True,
        )

    # ------------------------------------------------------------- desktop

    def desktop(self, title: str, body: str) -> None:
        try:
            if self.system == "Darwin":
                script = f'display notification {_osa(body)} with title {_osa(title)}'
                subprocess.run(["osascript", "-e", script], check=False, timeout=5,
                               capture_output=True)
            elif self.system == "Windows":
                ps = (
                    "[reflection.assembly]::loadwithpartialname('System.Windows.Forms');"
                    "$n=New-Object System.Windows.Forms.NotifyIcon;"
                    "$n.Icon=[System.Drawing.SystemIcons]::Information;$n.Visible=$true;"
                    f"$n.ShowBalloonTip(8000,'{_ps(title)}','{_ps(body)}',"
                    "[System.Windows.Forms.ToolTipIcon]::Info)"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False,
                               timeout=8, capture_output=True)
            elif shutil.which("notify-send"):
                subprocess.run(["notify-send", "-u", "critical", title, body], check=False,
                               timeout=5, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("desktop notification failed: %s", exc)

    # --------------------------------------------------------------- sound

    def sound(self, side: Optional[Side] = None) -> None:
        try:
            if self.system == "Darwin":
                name = "Glass" if side is Side.LONG else "Submarine"
                path = f"/System/Library/Sounds/{name}.aiff"
                subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return
            if self.system == "Windows":
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", "[console]::beep(880,250)"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            for player, args in (("paplay", ["/usr/share/sounds/freedesktop/stereo/message.oga"]),
                                 ("aplay", ["/usr/share/sounds/alsa/Front_Center.wav"])):
                if shutil.which(player):
                    subprocess.Popen([player, *args], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    return
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("sound failed: %s", exc)
        # Terminal bell is the universal fallback.
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass


def _osa(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ps(text: str) -> str:
    return text.replace("'", "''")
