"""valcoach — command line interface.

    valcoach init --riot-id Name#TAG --region eu --henrik-key KEY
    valcoach sync --count 10
    valcoach analyze --last 10 --html review.html
    valcoach deaths --last 5
    valcoach coach --last 10 --question "why do I lose so many 1v1s?"
    valcoach watch --interval 180 --coach
    valcoach demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .analysis.report import Report, build_report
from .config import Config, config_path, valcoach_home
from .maps import MapIndex, refresh_assets
from .models import Match
from .providers import PROVIDER_NAMES, ProviderError, iter_payloads, parse_payload
from .render.html import render_html
from .render.text import Painter, render_deaths, render_report, use_color
from .store import Store
from .watcher import Watcher, notify, review_window, sync_matches

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "demo_matches.json")
DEMO_NOTICE = (
    "This review was generated from synthetic matches bundled with valcoach, "
    "to show what the output looks like. It is not a record of real games."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_config(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    for key in ("riot_id", "region", "platform", "provider", "queue", "model",
                "db_path", "shard"):
        value = getattr(args, key, None)
        if value:
            overrides[key] = value
    if getattr(args, "henrik_key", None):
        overrides["henrik_api_key"] = args.henrik_key
    if getattr(args, "riot_key", None):
        overrides["riot_api_key"] = args.riot_key
    if getattr(args, "files", None):
        overrides["provider"] = "file"
        overrides["file_paths"] = list(args.files)
    return Config.load(overrides=overrides)


def _open_store(config: Config) -> Store:
    return Store(config.db_path)


def _resolve_target(config: Config, store: Store, player: str = "") -> str:
    needle = player or config.puuid or config.riot_id
    if not needle:
        raise ProviderError(
            "no player configured — run `valcoach init --riot-id Name#TAG`"
        )
    found = store.resolve_puuid(needle)
    if found:
        return found
    if needle == config.puuid:
        return needle
    raise ProviderError(
        f"no stored matches for {needle!r} — run `valcoach sync` first "
        f"(or `valcoach demo` to try it offline)"
    )


def _print_report(report: Report, args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return
    print(
        render_report(
            report,
            color=None if not getattr(args, "no_color", False) else False,
            detail=not getattr(args, "brief", False),
            death_limit=getattr(args, "death_limit", 25),
            show_deaths=not getattr(args, "no_deaths", False),
        )
    )


def _write_html(
    report: Report, path: str, notice: str = "", webfonts: bool = False
) -> str:
    path = os.path.expanduser(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_html(report, notice=notice, webfonts=webfonts))
    return path


def _maybe_coach(
    report: Report, config: Config, store: Store, args: argparse.Namespace
) -> None:
    """Attach a written review when asked for and possible."""
    from .coach import Coach, CoachUnavailable

    question = getattr(args, "question", None)
    focus = getattr(args, "focus", None)
    stream = not getattr(args, "json", False)
    coach = Coach(
        model=config.model, api_key=config.anthropic_api_key,
        effort=config.effort, stream=stream,
    )
    try:
        if stream:
            print()
            print("— your coach —\n", flush=True)
        result = coach.review(
            report,
            question=question,
            focus=focus,
            previous=store.recent_reports(report.puuid, limit=3),
            on_text=(lambda chunk: print(chunk, end="", flush=True)) if stream else None,
        )
        report.narrative = result.text
        if stream:
            print()
            if getattr(args, "verbose", False):
                print(
                    f"\n[{result.model}: {result.input_tokens} in / "
                    f"{result.output_tokens} out, "
                    f"{result.cache_read_tokens} cached]"
                )
    except CoachUnavailable as exc:
        print(f"\n(no written review: {exc})", file=sys.stderr)


def _save_report(store: Store, report: Report) -> int:
    return store.save_report(
        puuid=report.puuid,
        riot_id=report.riot_id,
        match_ids=[m.match_id for m in report.matches],
        metrics=report.metrics.summary(),
        findings=[f.to_dict() for f in report.findings],
        narrative=report.narrative,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if args.riot_id:
        config.riot_id = args.riot_id
    if not config.riot_id:
        print("A Riot ID is required: valcoach init --riot-id Name#TAG",
              file=sys.stderr)
        return 2

    if args.resolve and config.provider != "file":
        from .providers import build_provider

        try:
            provider = build_provider(config)
            puuid, canonical = provider.resolve_player(config.riot_id)
            config.puuid = puuid
            if canonical:
                config.riot_id = canonical
            print(f"resolved {config.riot_id} → {puuid}")
        except Exception as exc:  # noqa: BLE001 - keep setup usable offline
            print(f"could not resolve the Riot ID yet ({exc}).", file=sys.stderr)
            print("Config is still saved; `valcoach sync` will retry.",
                  file=sys.stderr)

    path = config.save()
    store = _open_store(config)
    print(f"config:   {path}")
    print(f"database: {config.db_path}")
    print(f"player:   {config.riot_id or '(unset)'}"
          + (f" [{config.puuid[:8]}…]" if config.puuid else ""))
    print(f"provider: {config.provider} · region {config.region}")
    keys = []
    if config.henrik_api_key:
        keys.append("henrik")
    if config.riot_api_key:
        keys.append("riot")
    if config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
        keys.append("anthropic")
    print(f"keys set: {', '.join(keys) if keys else 'none'}")
    print()
    print("Next: valcoach sync && valcoach analyze")
    store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    counts = store.counts()
    last_sync = store.get_meta("last_sync")
    print(f"valcoach {__version__}")
    print(f"home:      {valcoach_home()}")
    print(f"config:    {config_path()}"
          + ("" if os.path.exists(config_path()) else "  (not created yet)"))
    print(f"database:  {config.db_path}")
    print(f"player:    {config.riot_id or '(unset)'}")
    print(f"provider:  {config.provider} · region {config.region}"
          f" · platform {config.platform}")
    print(f"queue:     {config.queue or 'all'}")
    print(f"model:     {config.model}")
    print(
        "stored:    "
        + ", ".join(f"{v} {k}" for k, v in counts.items())
    )
    if last_sync:
        ago = int(time.time()) - int(last_sync)
        print(f"last sync: {ago // 60} min ago")
    maps = MapIndex(config.load_assets())
    print(
        "callouts:  "
        + ("loaded" if maps.has_callouts else "not downloaded (run `valcoach assets`)")
    )
    from .coach import Coach

    have_key = bool(config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    print(
        "coaching:  "
        + ("ready" if (Coach.available() and have_key)
           else "unavailable ("
                + ("install `anthropic`" if not Coach.available()
                   else "set ANTHROPIC_API_KEY")
                + ")")
    )
    store.close()
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    try:
        result = sync_matches(
            config, store, count=args.count, log=lambda msg: print(msg, flush=True)
        )
    except ProviderError as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        store.close()
        return 1
    print(
        f"fetched {result.fetched} matches, {result.new_count} new "
        f"({store.counts()['matches']} stored in total)"
    )
    for err in result.errors:
        print(f"  warning: {err}", file=sys.stderr)
    store.close()
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    added = seen = 0
    for path in args.paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            files = [
                os.path.join(expanded, name)
                for name in sorted(os.listdir(expanded))
                if name.endswith(".json")
            ]
        else:
            files = [expanded]
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    document = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  skipped {file_path}: {exc}", file=sys.stderr)
                continue
            documents = document if isinstance(document, list) else [document]
            for entry in documents:
                for payload in iter_payloads(entry):
                    match = parse_payload(payload)
                    if not match or not match.match_id:
                        continue
                    seen += 1
                    if store.save_match(match, payload, match.provider):
                        added += 1
    print(f"imported {seen} matches ({added} new)")
    store.close()
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    try:
        target = _resolve_target(config, store, args.player)
        report = review_window(
            config, store, last=args.last, puuid=target, strict=args.strict
        )
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        store.close()
        return 1

    if args.coach:
        _maybe_coach(report, config, store, args)
    _print_report(report, args)
    if args.html:
        print(
            f"\nHTML report: "
            f"{_write_html(report, args.html, webfonts=args.webfonts)}"
        )
    if not args.no_save:
        _save_report(store, report)
    store.close()
    return 0


def cmd_deaths(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    try:
        target = _resolve_target(config, store, args.player)
        report = review_window(config, store, last=args.last, puuid=target)
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        store.close()
        return 1

    deaths = report.deaths
    if args.map:
        deaths = [d for d in deaths if d.map_name.lower() == args.map.lower()]
    if args.side:
        deaths = [d for d in deaths if d.side == args.side]
    if args.untraded:
        deaths = [d for d in deaths if not d.traded]
    if args.first:
        deaths = [d for d in deaths if d.first_death_of_round]

    if args.json:
        print(json.dumps([asdict(d) for d in deaths], indent=2, default=str))
        store.close()
        return 0

    paint = Painter(use_color(False if args.no_color else None))
    print(paint.bold(f"{len(deaths)} deaths · {report.window}"))
    print(render_deaths(deaths, paint, limit=args.limit))
    m = report.metrics
    print()
    print(
        paint.dim(
            f"  untraded {m.untraded_death_rate}% · isolated "
            f"{m.isolated_death_rate}% · first death {m.first_death_rate}% of rounds "
            f"· avg at {m.avg_death_time_ms / 1000:.0f}s"
        )
    )
    if m.deaths_by_killer:
        top = ", ".join(f"{name} x{n}" for name, n in m.deaths_by_killer[:5])
        print(paint.dim(f"  killed most by: {top}"))
    if m.deaths_by_weapon:
        top = ", ".join(f"{name} x{n}" for name, n in m.deaths_by_weapon[:5])
        print(paint.dim(f"  killed most with: {top}"))
    store.close()
    return 0


def cmd_coach(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    try:
        target = _resolve_target(config, store, args.player)
        report = review_window(
            config, store, last=args.last, puuid=target, strict=args.strict
        )
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        store.close()
        return 1

    if not report.contexts:
        print("no matches to review — run `valcoach sync` first", file=sys.stderr)
        store.close()
        return 1

    _maybe_coach(report, config, store, args)
    if args.json:
        print(json.dumps(report.to_dict(include_deaths=False), indent=2, default=str))
    if args.html:
        print(
            f"\nHTML report: "
            f"{_write_html(report, args.html, webfonts=args.webfonts)}"
        )
    if not args.no_save:
        _save_report(store, report)
    store.close()
    return 0 if report.narrative else 1


def cmd_watch(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)

    def on_new(report: Report, result: Any) -> None:
        from .render.text import render_watch_line

        line = render_watch_line(report)
        print(f"\n{time.strftime('%H:%M')}  {line}")
        focus = report.focus(2)
        for f in focus:
            print(f"    · {f.title}: {f.summary}")
        if args.notify:
            notify("valcoach", line)
        if args.coach:
            _maybe_coach(report, config, store, args)
        if args.html:
            _write_html(report, args.html, webfonts=getattr(args, "webfonts", False))
        _save_report(store, report)

    watcher = Watcher(
        config, store, interval=args.interval, count=args.count,
        on_new_match=on_new,
    )
    try:
        watcher.run(max_cycles=args.cycles)
    finally:
        store.close()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on bundled synthetic matches — no key, no network."""
    config = Config.load(
        overrides={
            "db_path": args.db or os.path.join(valcoach_home(), "demo.db"),
            "riot_id": "You#0000",
            "provider": "file",
        }
    )
    store = _open_store(config)
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        payloads = json.load(handle)
    matches: List[Match] = []
    for payload in payloads:
        for single in iter_payloads(payload):
            match = parse_payload(single)
            if match:
                store.save_match(match, single, match.provider)
                matches.append(match)

    target = matches[0].find_player("You#0000")
    report = build_report(
        matches,
        puuid=target.ref.puuid,
        riot_id=target.ref.riot_id,
        trade_window_ms=config.trade_window_ms,
        strict=args.strict,
    )
    print(
        "Demo mode: these are synthetic matches bundled with valcoach, "
        "not real games.\n"
    )
    if args.coach:
        _maybe_coach(report, config, store, args)
    _print_report(report, args)
    if args.html:
        print(
            f"\nHTML report: "
            f"{_write_html(report, args.html, notice=DEMO_NOTICE, webfonts=args.webfonts)}"
        )
    store.close()
    return 0


def cmd_matches(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    target = None
    try:
        target = _resolve_target(config, store, args.player)
    except ProviderError:
        pass
    rows = store.match_rows(puuid=target, limit=args.limit)
    if not rows:
        print("no matches stored — run `valcoach sync` or `valcoach demo`")
        store.close()
        return 0
    print(f"{'when':<17}{'map':<10}{'queue':<14}{'rounds':>7}  match id")
    for row in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["started_at"] or 0))
        rounds = store.conn.execute(
            "SELECT COUNT(*) c FROM rounds WHERE match_id=?", (row["match_id"],)
        ).fetchone()["c"]
        print(
            f"{when:<17}{(row['map_name'] or '?'):<10}"
            f"{(row['queue'] or '?'):<14}{rounds:>7}  {row['match_id']}"
        )
    store.close()
    return 0


def cmd_assets(args: argparse.Namespace) -> int:
    try:
        counts = refresh_assets()
    except Exception as exc:  # noqa: BLE001 - network failure is the common case
        print(f"could not download map data: {exc}", file=sys.stderr)
        return 1
    print(
        "downloaded "
        + ", ".join(f"{v} {k}" for k, v in counts.items())
        + " (callout names will now appear in reports)"
    )
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    done = store.reindex()
    print(f"re-parsed {done} stored matches with the current analysers")
    store.close()
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    config = _load_config(args)
    store = _open_store(config)
    try:
        target = _resolve_target(config, store, args.player)
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        store.close()
        return 1
    reports = store.recent_reports(target, limit=args.limit)
    if not reports:
        print("no saved reviews yet — run `valcoach analyze`")
        store.close()
        return 0
    for item in reports:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(item["created_at"]))
        problems = [
            f for f in item["findings"] if f.get("severity") != "strength"
        ]
        print(f"\n{when} · {len(item['match_ids'])} matches · review #{item['id']}")
        metrics = item["metrics"]
        print(
            f"  K/D {metrics.get('kd')} · ADR {metrics.get('adr')} · "
            f"HS {metrics.get('hs_pct')}% · KAST {metrics.get('kast_pct')}%"
        )
        for f in problems[:3]:
            print(f"  · [{f.get('severity')}] {f.get('title')}")
    store.close()
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valcoach",
        description="Track your VALORANT matches and find out what you're doing wrong.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"valcoach {__version__}")

    def common(sub: argparse.ArgumentParser, player: bool = True) -> None:
        sub.add_argument("--provider", choices=PROVIDER_NAMES, help="data source")
        sub.add_argument("--region", help="na, eu, ap, kr, br, latam")
        sub.add_argument("--platform", help="pc or console")
        sub.add_argument("--shard", help="override the pd shard (local provider)")
        sub.add_argument("--queue", help="only this queue, e.g. competitive")
        sub.add_argument("--db-path", dest="db_path", help="SQLite file to use")
        sub.add_argument("--henrik-key", dest="henrik_key", help="HenrikDev API key")
        sub.add_argument("--riot-key", dest="riot_key", help="Riot API key")
        sub.add_argument("--files", nargs="+", help="read matches from JSON files")
        if player:
            sub.add_argument("--player", default="", help="Riot ID or puuid to analyse")

    def output(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--html", help="also write a standalone HTML report here")
        sub.add_argument("--no-color", action="store_true")
        sub.add_argument("--brief", action="store_true",
                         help="findings without the explanations")
        sub.add_argument("--no-deaths", action="store_true",
                         help="skip the death log")
        sub.add_argument("--death-limit", type=int, default=25)
        sub.add_argument("--webfonts", action="store_true",
                         help="load display fonts in the HTML report "
                              "(otherwise it needs no network at all)")

    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("init", help="save your Riot ID and keys")
    common(p, player=False)
    p.add_argument("--riot-id", dest="riot_id", help="Name#TAG")
    p.add_argument("--model", help="coaching model (default from config)")
    p.add_argument("--no-resolve", dest="resolve", action="store_false",
                   help="skip the puuid lookup")
    p.set_defaults(func=cmd_init, resolve=True)

    p = subs.add_parser("status", help="show configuration and what is stored")
    common(p, player=False)
    p.set_defaults(func=cmd_status)

    p = subs.add_parser("sync", help="download recent matches")
    common(p, player=False)
    p.add_argument("--count", type=int, default=10, help="matches to request")
    p.set_defaults(func=cmd_sync)

    p = subs.add_parser("import", help="ingest match JSON from disk")
    common(p, player=False)
    p.add_argument("paths", nargs="+", help="files or directories")
    p.set_defaults(func=cmd_import)

    p = subs.add_parser("analyze", help="full review of your recent matches")
    common(p)
    output(p)
    p.add_argument("--last", type=int, default=10, help="matches to include")
    p.add_argument("--strict", action="store_true", help="tighter benchmarks")
    p.add_argument("--coach", action="store_true",
                   help="also write the narrative review with Claude")
    p.add_argument("--question", help="ask the coach something specific")
    p.add_argument("--focus", help="weight the review, e.g. positioning")
    p.add_argument("--model", help="override the coaching model")
    p.add_argument("--no-save", action="store_true", help="do not store this review")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = subs.add_parser("deaths", help="every death, with what went wrong")
    common(p)
    p.add_argument("--last", type=int, default=5)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--map", help="filter to one map")
    p.add_argument("--side", choices=("attack", "defense"))
    p.add_argument("--untraded", action="store_true", help="only untraded deaths")
    p.add_argument("--first", action="store_true", help="only first deaths of a round")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_deaths)

    p = subs.add_parser("coach", help="written review from Claude")
    common(p)
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--question", help="ask something specific")
    p.add_argument("--focus", help="weight the review, e.g. economy")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--model", help="override the coaching model")
    p.add_argument("--json", action="store_true")
    p.add_argument("--html", help="also write an HTML report")
    p.add_argument("--webfonts", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_coach)

    p = subs.add_parser("watch", help="keep watching for new matches")
    common(p, player=False)
    p.add_argument("--interval", type=int, default=180, help="seconds between polls")
    p.add_argument("--count", type=int, default=5, help="matches to request per poll")
    p.add_argument("--coach", action="store_true", help="write a review per match")
    p.add_argument("--notify", action="store_true", help="desktop notification")
    p.add_argument("--html", help="rewrite this HTML report after each match")
    p.add_argument("--webfonts", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cycles", type=int, help="stop after this many polls")
    p.add_argument("--question", help=argparse.SUPPRESS)
    p.add_argument("--focus", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_watch)

    p = subs.add_parser("demo", help="try it on bundled synthetic matches")
    output(p)
    p.add_argument("--db", help="where to write the demo database")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--coach", action="store_true")
    p.add_argument("--question", help=argparse.SUPPRESS)
    p.add_argument("--focus", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = subs.add_parser("matches", help="list stored matches")
    common(p)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_matches)

    p = subs.add_parser("history", help="past reviews and how they changed")
    common(p)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_history)

    p = subs.add_parser("assets", help="download map callouts and name tables")
    p.set_defaults(func=cmd_assets)

    p = subs.add_parser("reindex", help="re-parse stored matches after an upgrade")
    common(p, player=False)
    p.set_defaults(func=cmd_reindex)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
