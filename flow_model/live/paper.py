"""Section 12's paper path: the committed engine, observed, never acted on.

This module turns a `MarketView` into a readable `PaperSignal` carrying every
field section 12 lists -- LONG/SHORT/WAIT, score, confidence, entry/stop/
target, R:R, regime, reasons, timestamp, data-quality status -- and it does so
by calling the SAME `SignalEngine.decide()` the backtester calls. It holds no
account, no connection and no way to act. The four things that shape it:

**1. It cannot place an order, and that is checkable rather than promised.**
The user's hardest rule is "do not deploy live trading or risk real money", so
it is enforced by the parse tree instead of by this paragraph. Two audits,
both callable, both expected to return an empty tuple:

* `audit_cannot_place_orders()` walks THIS FILE's AST and reports
  (a) any import whose root is not in `PERMITTED_IMPORT_ROOTS` -- an
  allowlist, not a denylist, because a denylist does not know tomorrow's
  broker SDK; (b) any import that is not at module level, since a
  function-local `import` would escape a reader's glance at the top of the
  file; (c) any IDENTIFIER one of whose underscore-separated segments is in
  `FORBIDDEN_IDENTIFIER_WORDS` -- `order`, `submit`, `send`, `connect`,
  `fill`, `position`, `trade`, `ledger` and the rest; (d) any call to a
  dynamic-access or clock builtin (`getattr`, `eval`, `open`, `now`...);
  (e) any `async def` or `await`, so there is no event loop to drive a
  socket. Matched on the AST, so the prose above cannot trip it and a string
  cannot hide from it, and on identifier SEGMENTS rather than substrings, so
  `is_tradable` is not mistaken for `trade`.

  Two consequences worth naming. The word list forbids `position` and
  `trade`, so this module physically cannot call
  `RiskLimitManager.on_position_opened` or `on_trade_closed`: the limit
  manager it is handed is read-only here, and a paper session cannot mutate
  the risk state a real run would. And `open` being forbidden leaves
  `Path.read_text` of this repo's own source -- what the audits below do --
  as the only filesystem access in the module.

  `session` is deliberately NOT a forbidden word (`PaperSession` needs it):
  a `requests.Session` is caught by the import allowlist, which is the check
  that owns that hazard.

* `audit_import_closure()` is the stronger one. It walks the TRANSITIVE
  closure of this module's `flow_model` imports across this repository's own
  source and reports any module in it that imports a network or
  process-spawning root (`socket`, `http`, `urllib`, `requests`, `asyncio`,
  `subprocess`, a named broker SDK...), or that lives under
  `flow_model.backtest`, which is where the order-submission machinery is.
  It returns an empty tuple today: 31 repository modules are reachable and
  not one of them can open a socket. What it does NOT cover is third-party
  code, so the honest scope is "this repository's own source"; the import
  allowlist is what bounds the third-party set, and it admits `numpy`,
  `pydantic` and the standard library's pure-computation modules only.

**2. One engine, not two.** `PaperSession` does not reimplement a gate, a
score or a stop. It calls `SignalEngine.decide(view, ...)` once per bar and
projects the resulting `EngineDecision`. Construction refuses an engine whose
`decide` or `evaluate` has been overridden by a subclass
(`type(engine).decide is not SignalEngine.decide`), which is the one way a
parallel implementation could have been slipped in behind the right type.
`tests/unit/test_paper.py` runs `BacktestLoop` and `PaperSession` over the
same `DataStore` and asserts the emitted signals are identical field for
field -- they are the same object graph, so they must be.

**3. n=0 is the live path, not an edge case.** At the committed config the
system takes no trades; over a year of synthetic NQ bars no bar's Flow Score
reaches the lowest setup threshold (PROVENANCE.md 2.4 and 4a hold the figures
of record). So every signal here is a WAIT, and a WAIT carries `None` for
entry, stop, target and R:R -- never `0.0`, which would read as a measured
geometry. `PaperSignal` validates that: directional signals carry the full
plan, WAITs carry none of it. Rates over observed bars are `MetricValue`s
from `analytics/metrics.py`, so a session that has observed nothing reports
`undefined[input_not_supplied]` rather than a zero, and there is no second
representation of absence in this file.

**4. The origin of the bars is on every signal.** No real market bar has ever
entered this system. A signal that printed `LONG NQ 18000.0` without saying
where the bar came from would read as a market observation, so every
`PaperSignal` carries `dataset_source`, `dataset_synthetic` and a non-empty
`origin_warning` unless the dataset positively declares real bars from a
named source. `origin_warning()` reads the same two fields of
`SymbolData.describe()` that `dashboard.build.origin_notice` reads, and the
default is suspicion. The dashboard's `DataOrigin` is the authoritative
taxonomy; it is not imported here only because the dashboard imports Plotly
and requirement 1's import-surface argument has to stay short enough to
read. A test pins the two to agree on when a warning is owed.

Determinism: no clock is read. Every timestamp on a `PaperSignal` comes from
the view that produced it, so replaying one dataset twice yields identical
signals.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from flow_model.analytics.metrics import MetricValue, Undefined
from flow_model.core.contracts import Signal
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    Regime,
    SetupType,
    Side,
    SignalAction,
    WaitReason,
)
from flow_model.core.model import FrozenModel
from flow_model.data.market_view import MarketView
from flow_model.data.quality import AvailabilityReport, QualityGrader
from flow_model.risk.limits import RiskLimitManager
from flow_model.signals.engine import EngineDecision, SignalEngine

__all__ = [
    "FORBIDDEN_CALL_NAMES",
    "FORBIDDEN_IDENTIFIER_WORDS",
    "NETWORK_ROOTS",
    "PERMITTED_IMPORT_ROOTS",
    "PaperError",
    "PaperSession",
    "PaperSessionSummary",
    "PaperSignal",
    "audit_cannot_place_orders",
    "audit_import_closure",
    "origin_warning",
]


class PaperError(RuntimeError):
    """A paper session was wired in a way that could not be observed safely."""


# ---------------------------------------------------------------------------
# The structural guards. Declared before anything they govern.
# ---------------------------------------------------------------------------

#: Every top-level import root this module may use. An ALLOWLIST: a denylist
#: of broker SDKs is a list of the ones that existed when it was written.
PERMITTED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"__future__", "ast", "collections", "pathlib", "typing", "pydantic", "flow_model"}
)

#: Underscore-separated identifier segments this module may not contain.
#: Matched per segment, never as a substring, so `is_tradable` is not a
#: `trade` and `reasons` is not a `send`.
FORBIDDEN_IDENTIFIER_WORDS: frozenset[str] = frozenset(
    {
        # placing one
        "order", "orders", "submit", "submitted", "submission", "place",
        "placed", "placement", "buy", "sell", "amend", "cancel", "modify",
        "liquidate", "flatten", "route", "venue",
        # a counterparty to place it with
        "broker", "brokerage", "execute", "execution", "exec", "transmit",
        "send", "sendall", "post", "request", "requests", "socket", "connect",
        "connection", "listen", "bind", "login", "logon", "auth",
        "authenticate", "credential", "credentials", "secret", "token",
        "apikey", "endpoint", "webhook", "url", "uri", "host", "port",
        # the state a real run would keep, and the modules that keep it
        "trade", "trades", "fill", "fills", "filled", "position", "positions",
        "portfolio", "ledger", "journal",
        # anything that leaves the process
        "write", "open", "save", "upload", "download",
    }
)

#: Calls that would defeat a static read of this file, plus every clock.
FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "__import__", "getattr", "setattr",
        "delattr", "globals", "locals", "vars", "input", "open",
        "now", "utcnow", "today", "time", "monotonic", "perf_counter", "sleep",
    }
)

#: Import roots that could reach a network or spawn a process. Used by the
#: closure audit over this repository's source, not as this file's own gate --
#: that is `PERMITTED_IMPORT_ROOTS`, which admits none of these by omission.
NETWORK_ROOTS: frozenset[str] = frozenset(
    {
        "socket", "socketserver", "ssl", "http", "urllib", "urllib3",
        "requests", "httpx", "aiohttp", "websocket", "websockets", "ftplib",
        "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "asyncio",
        "subprocess", "multiprocessing", "selectors", "webbrowser", "ctypes",
        "paramiko", "pika", "kafka", "redis", "zmq", "grpc",
        "ib_insync", "ibapi", "ibkr", "alpaca", "alpaca_trade_api", "ccxt",
        "oandapyV20", "tda", "schwab", "robin_stocks", "tradestation",
        "tastytrade", "polygon", "binance", "coinbase",
    }
)

#: Package prefix holding the order-submission machinery. Reaching it from
#: here would mean the paper path could construct a fill.
EXECUTION_PACKAGE: str = "flow_model.backtest"


def _identifier_names(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Every identifier in `tree`, with its line. Names, not string text."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.lineno, node.arg
        elif isinstance(node, ast.alias):
            yield 0, (node.asname or node.name.split(".")[-1])


def _called_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def audit_cannot_place_orders(source: str | None = None) -> tuple[str, ...]:
    """Every way this module departs from "it cannot act". Empty for this file.

    Checks the import allowlist, module-level-only imports, the forbidden
    identifier segments, the dynamic-access and clock builtins, and the
    absence of any coroutine. `source` is for the test that feeds it a
    deliberately bad module and expects the offence to be named.
    """
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    at_module_level: set[int] = {
        id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    offences: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            for root in roots:
                if root not in PERMITTED_IMPORT_ROOTS:
                    offences.append(
                        f"line {node.lineno}: import root {root!r} is not in "
                        "PERMITTED_IMPORT_ROOTS"
                    )
            if id(node) not in at_module_level:
                offences.append(
                    f"line {node.lineno}: import is not at module level, so a "
                    "reader of the file header would not see it"
                )
            if isinstance(node, ast.ImportFrom) and node.level:
                offences.append(
                    f"line {node.lineno}: relative import cannot be resolved by "
                    "the closure audit, so it may not be used here"
                )
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor)):
            offences.append(
                f"line {node.lineno}: coroutine machinery would give this module "
                "an event loop, which is what drives a socket"
            )
        elif isinstance(node, ast.Call):
            name = _called_name(node)
            if name in FORBIDDEN_CALL_NAMES:
                offences.append(f"line {node.lineno}: call to {name}()")

    for lineno, name in _identifier_names(tree):
        for segment in name.lower().split("_"):
            if segment in FORBIDDEN_IDENTIFIER_WORDS:
                offences.append(
                    f"line {lineno}: identifier {name!r} contains the forbidden "
                    f"segment {segment!r}"
                )
    return tuple(sorted(set(offences)))


def _repository_source(module: str, root: Path) -> Path | None:
    candidate = root / (module.replace(".", "/") + ".py")
    if candidate.is_file():
        return candidate
    candidate = root / module.replace(".", "/") / "__init__.py"
    return candidate if candidate.is_file() else None


def _repository_imports(module: str, root: Path) -> tuple[list[str], list[str]]:
    """(`flow_model` modules imported, non-repository roots imported)."""
    path = _repository_source(module, root)
    if path is None:
        return [], []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    internal: list[str] = []
    external: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                (internal if alias.name.startswith("flow_model") else external).append(
                    alias.name if alias.name.startswith("flow_model")
                    else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and not node.level:
            name = node.module or ""
            if name.startswith("flow_model"):
                internal.append(name)
                internal.extend(f"{name}.{alias.name}" for alias in node.names)
            else:
                external.append(name.split(".")[0])
    return internal, external


def audit_import_closure(root: Path | None = None) -> tuple[str, ...]:
    """Network or order-submission reachability from this module. Empty here.

    Walks the transitive closure of this module's `flow_model` imports over
    this repository's own source files and reports any module in it that
    imports a `NETWORK_ROOTS` member or lives under `EXECUTION_PACKAGE`.
    Third-party packages are out of scope and are bounded instead by
    `PERMITTED_IMPORT_ROOTS`; the scope is stated rather than implied.
    """
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    seeds, external = _repository_imports(__name__, base)
    offences: list[str] = [
        f"{__name__} imports {name!r}, a network root"
        for name in sorted(set(external))
        if name in NETWORK_ROOTS
    ]
    seen: set[str] = set()
    pending: list[str] = list(seeds)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        if _repository_source(module, base) is None:
            continue
        if module == EXECUTION_PACKAGE or module.startswith(f"{EXECUTION_PACKAGE}."):
            offences.append(
                f"{module} is reachable from {__name__}; {EXECUTION_PACKAGE} holds "
                "the machinery that turns an intent into a fill"
            )
        internal, outside = _repository_imports(module, base)
        offences.extend(
            f"{module} imports {name!r}, a network root" for name in sorted(set(outside))
            if name in NETWORK_ROOTS
        )
        pending.extend(internal)
    return tuple(sorted(set(offences)))


def reachable_repository_modules(root: Path | None = None) -> tuple[str, ...]:
    """The repository modules the closure audit covers, for the record."""
    base = Path(__file__).resolve().parents[2] if root is None else Path(root)
    seeds, _ = _repository_imports(__name__, base)
    seen: set[str] = set()
    pending = list(seeds)
    while pending:
        module = pending.pop()
        if module in seen or _repository_source(module, base) is None:
            seen.add(module)
            continue
        seen.add(module)
        internal, _ = _repository_imports(module, base)
        pending.extend(internal)
    return tuple(sorted(m for m in seen if _repository_source(m, base) is not None))


# ---------------------------------------------------------------------------
# Where the bars came from
# ---------------------------------------------------------------------------

#: Said on a signal whose dataset declared itself generated.
SYNTHETIC_WARNING: str = (
    "SYNTHETIC BAR. NOT A MARKET OBSERVATION. This signal was computed from a "
    "generated price series; no real market bar has ever entered this system."
)

#: Said when the dataset declared nothing, which is not the same as real.
UNDECLARED_WARNING: str = (
    "DATA ORIGIN UNDECLARED. The dataset behind this signal did not state "
    "whether its bars are real or generated, so it may not be read as a market "
    "observation. Every figure this project has published came from a "
    "synthetic generator."
)


def origin_warning(dataset_source: str, dataset_synthetic: bool | None) -> str:
    """The warning owed by these two declared fields, or `""` if none is.

    Reads the `source` and `synthetic` fields `SymbolData.describe()` emits,
    which is the same pair `dashboard.build.origin_notice` reads. Silence is
    `UNDECLARED_WARNING`, not an absence of one: only a dataset positively
    declaring `synthetic=False` from a named source gets `""`.
    """
    if dataset_synthetic:
        return SYNTHETIC_WARNING
    if (
        dataset_synthetic is False
        and dataset_source
        and dataset_source != "unknown"
    ):
        return ""
    return UNDECLARED_WARNING


# ---------------------------------------------------------------------------
# The emitted record
# ---------------------------------------------------------------------------


class PaperSignal(FrozenModel):
    """One bar's decision, as a reader should see it.

    Carries the engine's own `Signal` rather than a copy of its fields, so
    there is exactly one statement of the action, the score, the confidence,
    the regime, the reasons and the data-quality grade, and the parity test
    compares the contract object itself. The geometry, the feed detail and
    the origin are the fields `Signal` does not have room for.

    `planned_risk_dollars` is section 0.2's PLANNED unit,
    `|planned_entry - stop| * point_value * size`, carried off
    `EntryPlan.risk_dollars`, which the `TradeIntent` validator owns. It is
    not a third definition and nothing here recomputes it.
    """

    signal: Signal

    # --- the plan, or nothing. Never 0.0 on a WAIT. --------------------
    entry_price: float | None = Field(default=None, gt=0.0)
    stop_price: float | None = Field(default=None, gt=0.0)
    target_price: float | None = Field(default=None, gt=0.0)
    reward_risk: float | None = Field(default=None, gt=0.0)
    achievable_reward_risk: float | None = Field(default=None, ge=0.0)
    size: float | None = Field(default=None, gt=0.0)
    planned_risk_dollars: float | None = Field(default=None, gt=0.0)
    point_value: float | None = Field(default=None, gt=0.0)
    max_hold_bars: int | None = Field(default=None, gt=0)

    # --- the data-quality status, per section 12 -----------------------
    feed_quality: dict[Feed, DataQuality] = Field(default_factory=dict)
    overall_feed_quality: DataQuality = DataQuality.MISSING
    blocking_feeds: tuple[Feed, ...] = ()
    quality_note: str = ""

    # --- how much of the Flow Score had a feed at all ------------------
    available_points: float | None = Field(default=None, ge=0.0)
    total_points: float | None = Field(default=None, gt=0.0)
    incomputable_components: tuple[Component, ...] = ()
    availability_note: str = ""

    # --- where the bars came from --------------------------------------
    dataset_source: str = ""
    dataset_synthetic: bool | None = None
    origin_warning: str = UNDECLARED_WARNING

    # --- the audit trail -----------------------------------------------
    stages_run: tuple[str, ...] = ()
    blocking_stage: str = ""

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def ts(self):
        """The view's timestamp. No clock is read anywhere in this module."""
        return self.signal.ts

    @property
    def action(self) -> SignalAction:
        return self.signal.action

    @property
    def side(self) -> Side | None:
        return self.signal.side

    @property
    def setup(self) -> SetupType | None:
        return self.signal.setup

    @property
    def flow_score(self) -> float | None:
        return self.signal.flow_score

    @property
    def confidence(self) -> float:
        return self.signal.confidence

    @property
    def regime(self) -> Regime:
        return self.signal.regime

    @property
    def data_quality(self) -> DataQuality:
        return self.signal.data_quality

    @property
    def wait_reason(self) -> WaitReason | None:
        return self.signal.wait_reason

    @property
    def wait_detail(self) -> str:
        return self.signal.wait_detail

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.signal.reasons

    @property
    def component_points(self) -> dict[Component, float]:
        return self.signal.component_points

    @property
    def unavailable_points(self) -> float | None:
        """Declared rather than subtracted when either side is unmeasured."""
        if self.total_points is None or self.available_points is None:
            return None
        return round(self.total_points - self.available_points, 6)

    @property
    def needs_origin_warning(self) -> bool:
        return bool(self.origin_warning)

    @model_validator(mode="after")
    def _check(self) -> "PaperSignal":
        geometry = (
            self.entry_price,
            self.stop_price,
            self.target_price,
            self.reward_risk,
            self.size,
            self.planned_risk_dollars,
            self.point_value,
        )
        if self.signal.action is SignalAction.WAIT:
            if any(value is not None for value in geometry):
                raise PaperError(
                    f"{self.signal.symbol} @ {self.signal.ts.isoformat()}: a WAIT "
                    "carries no entry, stop, target or R:R. Reporting 0.0 for a "
                    "geometry that was never planned reads as a measurement"
                )
        elif any(value is None for value in geometry):
            raise PaperError(
                f"{self.signal.symbol} @ {self.signal.ts.isoformat()}: a "
                f"{self.signal.action.value} signal must carry the whole plan; "
                "a partial one cannot be read as a trade idea"
            )
        if self.origin_warning != origin_warning(
            self.dataset_source, self.dataset_synthetic
        ):
            raise PaperError(
                "origin_warning does not match the declared dataset fields "
                f"(source={self.dataset_source!r}, "
                f"synthetic={self.dataset_synthetic!r}); a signal may not "
                "understate where its bars came from"
            )
        return self

    def summary_lines(self) -> tuple[str, ...]:
        """The signal as a reader sees it. The origin warning comes first."""
        lines: list[str] = []
        if self.origin_warning:
            lines.append(self.origin_warning)
        head = (
            f"{self.symbol} {self.ts.isoformat()} {self.action.value}"
            f"  regime={self.regime.value}  data_quality={self.data_quality.value}"
        )
        lines.append(head)
        if self.flow_score is None:
            lines.append("  flow score: none computed at this bar")
        else:
            lines.append(
                f"  flow score: {self.flow_score:.6f}  "
                f"confidence={self.confidence:.6f}"
            )
        if self.action is SignalAction.WAIT:
            reason = "unstated" if self.wait_reason is None else self.wait_reason.value
            lines.append(f"  WAIT ({reason}): {self.wait_detail or 'no detail'}")
            lines.append("  entry/stop/target/R:R: none -- no plan was made")
        else:
            lines.append(
                f"  setup={self.setup.value if self.setup else 'none'}  "
                f"entry={self.entry_price}  stop={self.stop_price}  "
                f"target={self.target_price}  R:R={self.reward_risk}"
            )
            lines.append(
                f"  size={self.size}  planned risk={self.planned_risk_dollars} "
                f"(section 0.2 planned unit)"
            )
        if self.available_points is not None and self.total_points is not None:
            lines.append(
                f"  Flow Score points with a feed: {self.available_points:g} of "
                f"{self.total_points:g}"
                + (
                    "  unmeasured: "
                    + ", ".join(c.value for c in self.incomputable_components)
                    if self.incomputable_components
                    else ""
                )
            )
        elif self.availability_note:
            lines.append(f"  feed availability: {self.availability_note}")
        if self.blocking_feeds:
            lines.append(
                "  blocking feeds: " + ", ".join(f.value for f in self.blocking_feeds)
            )
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        return tuple(lines)


class PaperSessionSummary(FrozenModel):
    """Counts over the bars a session observed, and the rates they support.

    Rates are `MetricValue`s so a session that observed nothing says
    `undefined[input_not_supplied]` instead of `0.0`. `max_flow_score` is the
    one number here that answers the live question -- how far the best bar
    got towards the threshold -- and it is undefined when no bar was scored.
    """

    symbol: str
    bars_observed: int = Field(ge=0)
    by_action: dict[SignalAction, int] = Field(default_factory=dict)
    by_wait_reason: dict[WaitReason, int] = Field(default_factory=dict)
    scored_bars: int = Field(ge=0)
    directional_bars: int = Field(ge=0)
    scored_fraction: MetricValue
    directional_fraction: MetricValue
    max_flow_score: MetricValue
    first_ts: str = ""
    last_ts: str = ""
    origin_warning: str = UNDECLARED_WARNING

    def summary_lines(self) -> tuple[str, ...]:
        lines: list[str] = []
        if self.origin_warning:
            lines.append(self.origin_warning)
        lines.append(
            f"{self.symbol}: {self.bars_observed} bar(s) observed"
            + (f"  {self.first_ts} .. {self.last_ts}" if self.first_ts else "")
        )
        for action in SignalAction:
            lines.append(f"  {action.value:<6} {self.by_action.get(action, 0)}")
        for reason, count in sorted(
            self.by_wait_reason.items(), key=lambda kv: (-kv[1], kv[0].value)
        ):
            lines.append(f"    wait:{reason.value:<24} {count}")
        lines.append(f"  {self.scored_fraction}")
        lines.append(f"  {self.directional_fraction}")
        lines.append(f"  {self.max_flow_score}")
        lines.append(
            "  No instruction to act is produced by this module under any "
            "configuration."
        )
        return tuple(lines)


# ---------------------------------------------------------------------------
# The feed this session reads
# ---------------------------------------------------------------------------


class ViewFeed(Protocol):
    """The three methods a paper replay needs from a market source.

    Narrower than `backtest.loop.MarketSource` on purpose: that protocol also
    exposes the bars, quotes and intrabar slices the FILL machinery consumes,
    and depending on it would put `flow_model.backtest` in this module's
    import closure, which `audit_import_closure` exists to forbid.
    `StoreMarketSource` satisfies this one structurally, so the parity test
    hands equivalent feeds to both paths.
    """

    def clock_ns(self) -> Sequence[int]: ...

    def advance(self, ts_ns: int) -> None: ...

    def view(self) -> MarketView: ...


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class PaperSession:
    """Observes the committed engine bar by bar. Emits records, nothing else.

    `equity` and `limits` are the two stateful inputs `SignalEngine` takes by
    argument; they are passed straight through so the risk stage sees what a
    run would. They are read-only here: this module may not contain an
    identifier with the segment `position` or `trade`, so it cannot call
    `RiskLimitManager.on_position_opened` or `on_trade_closed` even by
    accident, and a paper session therefore cannot move the risk state.
    """

    def __init__(
        self,
        config,
        symbol: str,
        *,
        engine: SignalEngine | None = None,
        calendar=None,
        availability: AvailabilityReport | None = None,
        dataset: Mapping[str, object] | None = None,
        equity: float | None = None,
        limits: RiskLimitManager | None = None,
    ) -> None:
        self.engine = self._shared_engine(config, symbol, engine, calendar)
        if self.engine.symbol != symbol:
            raise PaperError(
                f"the engine is built for {self.engine.symbol!r} but this session "
                f"is for {symbol!r}; the feature computers carry that symbol's "
                "InstrumentSpec"
            )
        self.config = config
        self.symbol = symbol
        self.availability = (
            availability if availability is not None else self.engine.availability
        )
        if self.availability is not None and self.availability.symbol != symbol:
            raise PaperError(
                f"the availability report is for {self.availability.symbol!r} but "
                f"this session is for {symbol!r}"
            )
        declared: Mapping[str, object] = dataset if dataset else {}
        raw = declared.get("source")
        self.dataset_source = str(raw) if raw else ""
        marker = declared.get("synthetic")
        self.dataset_synthetic = None if marker is None else bool(marker)
        self.origin_warning = origin_warning(
            self.dataset_source, self.dataset_synthetic
        )
        self.equity = None if equity is None else float(equity)
        self.limits = limits
        self._signals: list[PaperSignal] = []
        self._latest_ns: int | None = None

    @staticmethod
    def _shared_engine(config, symbol, engine, calendar) -> SignalEngine:
        """The backtest's engine, or a refusal. Section 12's "identical object".

        A subclass that overrode `decide` or `evaluate` would satisfy an
        `isinstance` check and still be a parallel implementation, which is
        the exact failure mode section 12 names, so the bound methods are
        compared against `SignalEngine`'s own.
        """
        if engine is None:
            return SignalEngine(config, symbol, calendar=calendar)
        if not isinstance(engine, SignalEngine):
            raise PaperError(
                f"a paper session runs the backtest's SignalEngine; got "
                f"{type(engine).__name__}. A second implementation of the "
                "pipeline would drift from the backtest silently"
            )
        for name in ("decide", "decide_from", "evaluate"):
            if getattr(type(engine), name) is not getattr(SignalEngine, name):
                raise PaperError(
                    f"{type(engine).__name__} overrides SignalEngine.{name}, so "
                    "the paper path would not run the pipeline the backtest runs"
                )
        if calendar is not None and engine.calendar is not calendar:
            raise PaperError(
                "the engine was built with a different session calendar than the "
                "one handed to this session; one dataset has one calendar"
            )
        return engine

    # --- observing ------------------------------------------------------

    @property
    def signals(self) -> tuple[PaperSignal, ...]:
        return tuple(self._signals)

    def observe(self, view: MarketView) -> PaperSignal:
        """The `PaperSignal` at `view.now`, from one `decide()` call."""
        if view.symbol != self.symbol:
            raise PaperError(
                f"this session is for {self.symbol!r} and was handed a view for "
                f"{view.symbol!r}"
            )
        stamp = int(view._now_ns) if hasattr(view, "_now_ns") else None
        stamp = stamp if stamp is not None else int(view.now.timestamp())
        if self._latest_ns is not None and stamp <= self._latest_ns:
            raise PaperError(
                f"{self.symbol}: this session has already observed "
                f"{self._latest_ns}, and was handed {stamp}. A paper feed is "
                "point-in-time: re-reading a bar it has passed would let a later "
                "signal be computed from an earlier instant"
            )
        self._latest_ns = stamp
        decision = self.engine.decide(view, equity=self.equity, limits=self.limits)
        record = self._record(decision)
        self._signals.append(record)
        return record

    def replay(self, views: Iterable[MarketView]) -> tuple[PaperSignal, ...]:
        """Observe every view in order. Returns only this call's signals."""
        start = len(self._signals)
        for view in views:
            self.observe(view)
        return tuple(self._signals[start:])

    def replay_feed(self, feed: ViewFeed) -> tuple[PaperSignal, ...]:
        """Observe a `ViewFeed` across its own clock, in clock order."""
        start = len(self._signals)
        for ts_ns in feed.clock_ns():
            feed.advance(int(ts_ns))
            self.observe(feed.view())
        return tuple(self._signals[start:])

    # --- projection -----------------------------------------------------

    def _record(self, decision: EngineDecision) -> PaperSignal:
        """Project an `EngineDecision` onto a `PaperSignal`. Nothing recomputed."""
        plan = decision.plan
        quality = decision.quality
        available, total, incomputable, note = self._availability(decision)
        blocking = decision.blocking
        return PaperSignal(
            signal=decision.signal,
            entry_price=None if plan is None else plan.entry_price,
            stop_price=None if plan is None else plan.stop_price,
            target_price=None if plan is None else plan.target_price,
            reward_risk=None if plan is None else plan.planned_r_multiple,
            achievable_reward_risk=None if plan is None else plan.achievable_rr,
            size=None if plan is None else plan.size,
            planned_risk_dollars=None if plan is None else plan.risk_dollars,
            point_value=None if plan is None else plan.point_value,
            max_hold_bars=None if plan is None else plan.max_hold_bars,
            feed_quality=(
                {}
                if quality is None
                else {feed: status.quality for feed, status in quality.statuses.items()}
            ),
            overall_feed_quality=(
                DataQuality.MISSING if quality is None else quality.overall
            ),
            blocking_feeds=() if quality is None else tuple(quality.blocking_feeds),
            quality_note="" if quality is None else quality.note,
            available_points=available,
            total_points=total,
            incomputable_components=incomputable,
            availability_note=note,
            dataset_source=self.dataset_source,
            dataset_synthetic=self.dataset_synthetic,
            origin_warning=self.origin_warning,
            stages_run=tuple(stage.value for stage in decision.stages),
            blocking_stage="" if blocking is None else blocking.stage.value,
        )

    def _availability(
        self, decision: EngineDecision
    ) -> tuple[float | None, float | None, tuple[Component, ...], str]:
        """How much of the Flow Score had a feed, and where that was read.

        The decision's own `FlowScoreOutcome` wins when the pipeline reached
        the Flow Score stage, because those are the numbers the decision
        actually used. On the live path it does not reach that stage -- the
        data-quality gate rejects a bars-only feed first -- so the session's
        `AvailabilityReport` answers, and when neither exists the shortfall is
        reported as unmeasured rather than as a full 100.
        """
        outcome = decision.flow
        if outcome is not None:
            return (
                outcome.available_points,
                outcome.max_points,
                tuple(
                    component
                    for component, score in outcome.components.items()
                    if not score.available
                ),
                "from this bar's Flow Score stage",
            )
        report = self.availability
        if report is not None:
            return (
                report.available_points,
                report.total_points,
                tuple(report.incomputable_components),
                "from the dataset's feed availability report",
            )
        return (
            None,
            None,
            (),
            "not measured: no AvailabilityReport was supplied and this bar did "
            "not reach the Flow Score stage",
        )

    # --- the session's own statement ------------------------------------

    def summary(self) -> PaperSessionSummary:
        """Counts and rates over what this session has observed so far."""
        emitted = self._signals
        n = len(emitted)
        by_action: dict[SignalAction, int] = {}
        by_wait: dict[WaitReason, int] = {}
        scores: list[float] = []
        directional = 0
        for record in emitted:
            by_action[record.action] = by_action.get(record.action, 0) + 1
            if record.wait_reason is not None:
                by_wait[record.wait_reason] = by_wait.get(record.wait_reason, 0) + 1
            if record.flow_score is not None:
                scores.append(record.flow_score)
            if record.action is not SignalAction.WAIT:
                directional = directional + 1
        return PaperSessionSummary(
            symbol=self.symbol,
            bars_observed=n,
            by_action=by_action,
            by_wait_reason=by_wait,
            scored_bars=len(scores),
            directional_bars=directional,
            scored_fraction=self._fraction("scored_bar_fraction", len(scores), n),
            directional_fraction=self._fraction(
                "directional_bar_fraction", directional, n
            ),
            max_flow_score=(
                MetricValue.measured(
                    "max_flow_score", max(scores), n=len(scores), unit="points"
                )
                if scores
                else MetricValue.undefined_because(
                    "max_flow_score",
                    Undefined.INPUT_NOT_SUPPLIED,
                    n=0,
                    unit="points",
                    note=(
                        "no observed bar reached the Flow Score stage, so there is "
                        "no score to take a maximum over"
                    ),
                )
            ),
            first_ts=emitted[0].ts.isoformat() if emitted else "",
            last_ts=emitted[-1].ts.isoformat() if emitted else "",
            origin_warning=self.origin_warning,
        )

    @staticmethod
    def _fraction(name: str, count: int, n: int) -> MetricValue:
        if n == 0:
            return MetricValue.undefined_because(
                name,
                Undefined.INPUT_NOT_SUPPLIED,
                n=0,
                unit="fraction",
                note="this session has not observed a bar",
            )
        return MetricValue.measured(name, count / n, n=n, unit="fraction")

    # --- wiring ---------------------------------------------------------

    @classmethod
    def from_store(
        cls,
        config,
        symbol: str,
        store,
        *,
        engine: SignalEngine | None = None,
        calendar=None,
        equity: float | None = None,
        limits: RiskLimitManager | None = None,
    ) -> "PaperSession":
        """Wire a session from a `DataStore`, origin declaration included.

        One object answers both questions a signal must carry: the feed
        availability report (the 55-of-100 the brief names, measured by
        `QualityGrader`, not asserted here) and the dataset's own `source` /
        `synthetic` declaration. Taking them from anywhere else is how a
        synthetic signal ends up reading as a market observation.
        """
        data = store.get(symbol)
        grader = QualityGrader(config.data, config.flow_score)
        return cls(
            config,
            symbol,
            engine=engine,
            calendar=calendar,
            availability=grader.availability(data),
            dataset=data.describe(),
            equity=equity,
            limits=limits,
        )
