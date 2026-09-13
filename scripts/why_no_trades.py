#!/usr/bin/env python3
"""Answer "why isn't the bot trading?" from what actually happened.

    python scripts/why_no_trades.py

Reads the bot's own database and config and walks the funnel from the top:
config sanity -> is anything being discovered -> where candidates are dying
-> what the risk gates rejected. Prints the single most likely cause at the
end rather than making you correlate five tables by hand.

This exists because "the bot never trades" and "the market had no good
setups" produce identical logs. Everything below is read-only; running it
never places or cancels anything.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.data.candles import Timeframe  # noqa: E402
from app.scanner.loop import scanner_blocked_reason  # noqa: E402
from app.startup_checks import check_config_coherence  # noqa: E402

BAR = "=" * 72


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        diagnose(db)
    finally:
        db.close()


def diagnose(db) -> None:
    causes: list[str] = []

    print(BAR)
    print("  WHY ISN'T THE BOT TRADING?")
    print(BAR)

    # ---- 1. config sanity -------------------------------------------------
    print("\n[1/5] Config coherence")
    warnings = check_config_coherence()
    if warnings:
        for w in warnings:
            print(f"      !! {w}")
        causes.append("config warnings above")
    else:
        print("      OK - no impossible combinations")

    try:
        tf = Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
        need_h = settings.SIGNAL_SCORE_MIN_CANDLES * tf.seconds / 3600
        print(f"      score gate: >= {settings.MIN_SIGNAL_SCORE_TO_ENTER:.0f}/100, "
              f"needs {settings.SIGNAL_SCORE_MIN_CANDLES} x {tf.value} ({need_h:.0f}h of history)")
    except ValueError:
        pass

    # ---- 2. is the bot halted? -------------------------------------------
    print("\n[2/5] Trading halt")
    from app.risk.manager import is_trading_halted
    from app.state import get_state

    if is_trading_halted(db):
        reason = get_state(db, "trading_halt_reason", "") or "unknown"
        print(f"      !! HALTED: {reason}")
        print("      Resume from the dashboard, or POST /api/resume.")
        causes.append("trading is halted")
    else:
        print("      OK - not halted")

    # ---- 3. is anything being found? -------------------------------------
    print("\n[3/5] Token discovery")
    blocked = scanner_blocked_reason()
    scanned = db.query(models.ScannedToken).count()
    signals = db.query(models.Signal).count()

    if blocked:
        print(f"      !! scanner not running: {blocked}")
        causes.append("scanner disabled")
    else:
        print(f"      scanner is enabled (every {settings.SCANNER_INTERVAL_SECONDS}s)")

    print(f"      tokens ever discovered: {scanned}")
    print(f"      signals ever recorded:  {signals}")
    if scanned == 0 and not blocked:
        print("      !! nothing discovered yet. Either it hasn't run a cycle,")
        print("         or the discovery APIs returned nothing usable.")
        print("         Run: python scripts/scan_once.py")
        causes.append("no tokens discovered")

    # ---- 4. where candidates die -----------------------------------------
    print("\n[4/5] Scanner funnel (where candidates stop)")
    stages = Counter(
        row.last_stage or "unknown" for row in db.query(models.ScannedToken).all()
    )
    if stages:
        for stage, count in stages.most_common():
            print(f"      {stage:<12} {count:>5}")
        reasons = Counter(
            (row.last_reason or "")[:60]
            for row in db.query(models.ScannedToken)
            .filter(models.ScannedToken.last_stage == "prescreen").all()
        )
        if reasons:
            print("\n      Top pre-screen rejection reasons:")
            for reason, count in reasons.most_common(5):
                print(f"        {count:>4}x  {reason}")
        if stages.get("traded", 0) == 0 and stages.get("prescreen", 0) > 0:
            causes.append("everything is failing the pre-screen (thresholds too strict?)")
    else:
        print("      (no scanned tokens yet)")

    # ---- 5. what the gates rejected --------------------------------------
    print("\n[5/5] Risk / gate rejections")
    events = Counter(e.event_type for e in db.query(models.RiskEvent).all())
    if events:
        for event_type, count in events.most_common():
            print(f"      {event_type:<28} {count:>5}")
        if events.get("signal_score_rejected"):
            causes.append(
                f"the signal score gate rejected {events['signal_score_rejected']} candidate(s) - "
                f"MIN_SIGNAL_SCORE_TO_ENTER={settings.MIN_SIGNAL_SCORE_TO_ENTER:.0f} may be too high"
            )
        if events.get("signal_score_unavailable"):
            causes.append(
                f"{events['signal_score_unavailable']} candidate(s) had no usable candle data - "
                "GeckoTerminal may be unreachable, rate limited, or the token too new"
            )
        if events.get("rug_check_rejected"):
            causes.append(f"the rug check rejected {events['rug_check_rejected']} candidate(s)")
    else:
        print("      (no risk events recorded)")

    trades = db.query(models.Trade).filter_by(side="buy", status="filled").count()
    print(f"\n      positions actually opened: {trades}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + BAR)
    if trades > 0:
        print(f"  The bot HAS traded ({trades} position(s) opened). Nothing is broken.")
    elif causes:
        print("  MOST LIKELY CAUSE(S), in order:")
        for i, cause in enumerate(causes, 1):
            print(f"    {i}. {cause}")
    else:
        print("  No obvious blocker found. The bot may simply not have run a full")
        print("  cycle yet, or genuinely found no qualifying setups. Let it run")
        print("  and check again in an hour.")
    print(BAR)


if __name__ == "__main__":
    main()
