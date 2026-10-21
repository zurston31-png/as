"""`dashboard/build.py`: the panels, and the four refusals that shape them.

Section 13's gate for Phase 10 is "renders from real run artifacts", so the
first test in this file runs the real configured engine over real (synthetic)
bars and renders the page from the resulting `RunResult`. That run takes zero
trades -- the live path -- so the rest of the zero-trade tests are assertions
about a genuine artifact rather than about a constructed one.

Where a POPULATED run is needed, the trades are constructed. Nothing here
moves a threshold, a weight or a config default to produce them: at the
committed config no bar's Flow Score reaches the lowest setup threshold, and
lowering one to reach the populated rendering path is the move the brief
forbids. The constructed trades fabricate no claim -- they exist to prove that
the page prints exactly the digits `analytics` produced.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from flow_model.analytics.metrics import (
    AnalyticsError,
    Undefined,
    performance_summary,
    win_rate,
)
from flow_model.backtest.loop import (
    BacktestLoop,
    RunProvenance,
    RunResult,
    StoreMarketSource,
)
from flow_model.config.loader import default_config
from flow_model.core.contracts import EquityPoint
from flow_model.core.enums import Regime, SplitPhase
from flow_model.dashboard.build import (
    BANNER_MARKER,
    DashboardError,
    DashboardInputs,
    DashboardPage,
    DataOrigin,
    Panel,
    PanelId,
    assert_origin_is_stated,
    audit_no_local_statistics,
    build_html,
    build_page,
    collect,
    origin_notice,
    render,
    write_dashboard,
)
from flow_model.monte_carlo.engine import (
    MonteCarloPlan,
    MonteCarloRefusal,
    run_monte_carlo,
)
from flow_model.data import (
    DataStore,
    SyntheticAdapter,
    SyntheticConfig,
    TradingCalendar,
)
from flow_model.validation.splits import DataAccessGuard, DateRange, SealAudit
from flow_model.validation.walk_forward import (
    WindowSpec,
    evaluate_folds,
    generate_folds,
)

T0 = datetime(2020, 6, 1, 10, 0, tzinfo=timezone.utc)
START_EQUITY = 100_000.0


# ---------------------------------------------------------------------------
# Section 13's gate: a real run of the real engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_empty_run() -> RunResult:
    """A real `BacktestLoop` run over synthetic 2016 NQ bars.

    The committed default config, UNCHANGED, and a window three weeks long --
    far from the sealed 2023-01-01..2025-01-01 range, which is neither
    requested nor opened.
    """
    config = default_config()
    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(),
        seed=config.seed,
        spec_by_symbol=dict(config.instruments),
        interval_seconds=config.data.primary_interval_seconds,
        calendar=calendar,
    )
    store = DataStore(latency_seconds=config.data.data_latency_seconds)
    store.load(
        adapter,
        "NQ",
        date(2016, 1, 4),
        date(2016, 1, 25),
        primary_interval=config.data.primary_interval_seconds,
    )
    loop = BacktestLoop(config, "NQ", calendar=calendar, git_commit="", git_dirty=False)
    return loop.run(StoreMarketSource.from_config(store, "NQ", config))


@pytest.fixture(scope="module")
def real_page(real_empty_run) -> DashboardPage:
    return build_page(collect(real_empty_run))


class TestRendersFromRealRunArtifacts:
    """Phase 10's gate, and the zero-trade behaviour it exposes."""

    def test_the_real_run_took_no_trades_and_still_produced_an_equity_series(
        self, real_empty_run
    ):
        """The premise of every assertion below. If this ever changes, the
        zero-trade expectations in this file are measuring the wrong thing."""
        assert real_empty_run.trade_count == 0
        assert len(real_empty_run.equity) > 1
        assert real_empty_run.provenance.dataset["synthetic"] is True

    def test_all_eleven_panels_render_from_the_run_artifact(self, real_page):
        assert tuple(p.panel_id for p in real_page.panels) == tuple(PanelId)
        assert len(real_page.panels) == 11

    def test_every_trade_outcome_panel_is_undefined_with_no_trades(self, real_page):
        for panel_id in (
            PanelId.EQUITY_CURVE,
            PanelId.DRAWDOWN_CURVE,
            PanelId.DAILY_PNL,
            PanelId.CUMULATIVE_PNL,
            PanelId.FLOW_SCORE_DISTRIBUTION,
            PanelId.REGIME_PERFORMANCE,
            PanelId.TRADE_DISTRIBUTION,
        ):
            panel = real_page.panel(panel_id)
            assert panel.undefined is Undefined.NO_TRADES, panel_id
            assert panel.body == ""
            assert panel.n == 0

    def test_no_flat_equity_curve_is_drawn_at_the_starting_balance(self, real_page):
        """The specifically forbidden rendering.

        The equity series EXISTS and is flat at the starting balance, so the
        tempting thing is to plot it. A flat line at 100,000 reads as "we
        traded and ended level"; the truth is that nothing was measured.
        """
        page = render(real_page, inline_plotlyjs=False)
        equity = real_page.panel(PanelId.EQUITY_CURVE)
        assert equity.body == ""
        assert "undefined, n=0" in page
        assert f"{START_EQUITY:g}" not in equity.body
        # No Plotly div was emitted for any undefined panel.
        for panel in real_page.panels:
            if not panel.is_defined:
                assert f'id="{panel.panel_id.value}"' not in page

    def test_the_page_states_where_the_trades_went(self, real_page):
        """On an empty run the funnel is the whole story, and it is counts."""
        stages = dict(real_page.funnel)
        assert "wait_reasons" in stages
        assert "blocking_stages" in stages
        assert any(key.startswith("terminal gate") for key in stages)

    def test_rendering_the_same_run_twice_is_byte_identical(self, real_empty_run):
        first = build_html(real_empty_run, inline_plotlyjs=False)
        second = build_html(real_empty_run, inline_plotlyjs=False)
        assert first == second

    def test_the_page_carries_no_render_clock_reading(self, real_page, real_empty_run):
        """A timestamp from the clock would make two renders of one run
        differ, so the only timestamp allowed is the run's own."""
        facts = dict(real_page.facts)
        stamp = facts["run timestamp (from the run, not the clock)"]
        created = real_empty_run.provenance.created_at
        if created is None:
            assert "not recorded" in stamp
        else:
            assert stamp == created.isoformat()

    def test_the_page_is_self_contained_when_asked(self, real_page):
        assert "<script>" in render(real_page, inline_plotlyjs=True)
        assert "not inlined" in render(real_page, inline_plotlyjs=False)

    def test_write_dashboard_writes_the_page(self, real_page, tmp_path):
        target = write_dashboard(
            render(real_page, inline_plotlyjs=False), tmp_path / "sub" / "run.html"
        )
        assert target.is_file()
        assert target.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
# The origin of the data, which the page may not soften
# ---------------------------------------------------------------------------


class TestDataOrigin:
    def test_a_declared_synthetic_dataset_is_warned_about(self):
        notice = origin_notice({"source": "synthetic_v1", "synthetic": True})
        assert notice.origin is DataOrigin.SYNTHETIC
        assert notice.needs_warning is True
        assert "NOT A MARKET RESULT" in notice.headline

    @pytest.mark.parametrize(
        "dataset",
        [
            None,
            {},
            {"source": "unknown"},
            {"source": ""},
            {"synthetic": False},
            {"source": "unknown", "synthetic": False},
        ],
    )
    def test_silence_earns_the_warning(self, dataset):
        """The default is suspicion. "No marker" and "real bars" are not the
        same claim, and only one of them has been checked."""
        notice = origin_notice(dataset)
        assert notice.origin is DataOrigin.UNDECLARED
        assert notice.needs_warning is True
        assert "NOT EVIDENCE" in notice.watermark

    def test_only_a_positive_declaration_renders_as_real(self):
        notice = origin_notice({"source": "databento_nq", "synthetic": False})
        assert notice.origin is DataOrigin.REAL
        assert notice.needs_warning is False
        assert "databento_nq" in notice.watermark

    def test_the_banner_is_present_on_a_synthetic_page(self, real_page):
        page = render(real_page, inline_plotlyjs=False)
        assert BANNER_MARKER in page
        assert "SYNTHETIC DATA" in page

    def test_every_panel_carries_the_origin_in_its_own_provenance(self, real_page):
        """A panel screenshotted alone must still say where its data came from."""
        for panel in real_page.panels:
            assert panel.provenance.strip()
            assert "synthetic=True" in panel.provenance, panel.panel_id

    def test_the_origin_is_drawn_inside_every_figure(self, populated_page):
        """A cropped IMAGE of a chart carries the warning too, because the
        annotation is part of the figure rather than part of the page."""
        page = render(populated_page, inline_plotlyjs=False)
        drawn = [p for p in populated_page.panels if p.is_defined and "plotly" in p.body]
        assert drawn
        for panel in drawn:
            assert populated_page.notice.watermark in panel.body, panel.panel_id
        assert page.count(populated_page.notice.watermark) > len(populated_page.panels)

    def test_the_real_rendering_passes_its_own_origin_check(self, real_page):
        assert_origin_is_stated(render(real_page, inline_plotlyjs=False), real_page)

    def test_a_document_without_the_banner_is_refused(self, real_page):
        """The postcondition, exercised against a document that lost it.

        The check reads the rendered TEMPLATE rather than a constant it shares
        with the emitter, so it can actually fail -- which is the only reason
        it is worth having.
        """
        with pytest.raises(DashboardError, match="no origin banner"):
            assert_origin_is_stated("<html><body>charts</body></html>", real_page)

    def test_a_document_missing_a_panel_ribbon_is_refused(self, real_page):
        """One ribbon per panel, because the ribbon is what survives a crop."""
        one_ribbon = f'<div id="{BANNER_MARKER}"></div><div class="ribbon"></div>'
        with pytest.raises(DashboardError, match="origin ribbon"):
            assert_origin_is_stated(one_ribbon, real_page)

    def test_a_real_data_page_needs_neither(self, real_page):
        """A positively declared real dataset is the one case that does not
        get the warning, so the check returns early rather than raising."""
        real = real_page.replace(
            notice=origin_notice({"source": "databento_nq", "synthetic": False})
        )
        assert_origin_is_stated("<html></html>", real)

    def test_a_hostile_dataset_source_is_escaped(self):
        notice = origin_notice(
            {"source": "<script>alert(1)</script>", "synthetic": True}
        )
        page = render(
            DashboardPage(
                notice=notice,
                panels=tuple(
                    Panel(
                        panel_id=panel_id,
                        title="t",
                        provenance="p",
                        undefined=Undefined.NO_TRADES,
                    )
                    for panel_id in PanelId
                ),
                run_id="r",
                symbol="NQ",
                facts=(),
                funnel=(),
            ),
            inline_plotlyjs=False,
        )
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page


# ---------------------------------------------------------------------------
# Structural refusals
# ---------------------------------------------------------------------------


class TestPanelIsAMetricValueShape:
    def test_a_panel_carrying_both_a_body_and_a_reason_is_refused(self):
        with pytest.raises(ValidationError, match="both a body and an undefined"):
            Panel(
                panel_id=PanelId.EQUITY_CURVE,
                title="t",
                provenance="p",
                body="<div></div>",
                undefined=Undefined.NO_TRADES,
            )

    def test_a_panel_carrying_neither_is_refused(self):
        with pytest.raises(ValidationError, match="neither"):
            Panel(panel_id=PanelId.EQUITY_CURVE, title="t", provenance="p")

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_a_panel_without_a_provenance_label_is_refused(self, blank):
        with pytest.raises(ValidationError, match="no provenance label"):
            Panel(
                panel_id=PanelId.EQUITY_CURVE,
                title="t",
                provenance=blank,
                undefined=Undefined.NO_TRADES,
            )

    def test_a_page_may_not_drop_a_panel(self, real_page):
        kept = tuple(
            p for p in real_page.panels if p.panel_id is not PanelId.DRAWDOWN_CURVE
        )
        with pytest.raises(ValidationError, match="section 12 names"):
            real_page.replace(panels=kept)

    def test_a_page_may_not_reorder_its_panels(self, real_page):
        shuffled = (real_page.panels[1], real_page.panels[0], *real_page.panels[2:])
        with pytest.raises(ValidationError, match="in that order"):
            real_page.replace(panels=shuffled)


class TestNoStatisticIsComputedHere:
    def test_the_module_contains_no_arithmetic_and_no_numeric_import(self):
        """The guarantee, as a property of the parse tree.

        If this fails, the offending line is in the message: either compute
        the number in `analytics/` and read the `MetricValue`, or the page can
        disagree with the report and nothing will say which is wrong.
        """
        assert audit_no_local_statistics() == ()

    @pytest.mark.parametrize(
        "source, should_trip",
        [
            ("rate = wins / n", True),
            ("pct = fraction * 100", True),
            ("total = a + b", True),
            ("x -= 1", True),
            ("peak = max(values)", True),
            ("depth = abs(low)", True),
            ("shown = round(value, 2)", True),
            ("import numpy as np", True),
            ("from math import sqrt", True),
            ("import statistics", True),
            # The shapes the module actually uses, which must NOT trip.
            ('value: MetricValue | None = None', False),
            ('text = f"{v.value:.6g} {v.unit}"', False),
            ('joined = ", ".join(parts)', False),
            ("count = len(rows)", False),
            ("ordered = sorted(keys)", False),
            ('"""a docstring mentioning a win rate of wins / n"""', False),
        ],
    )
    def test_the_audit_trips_on_arithmetic_and_not_on_formatting(
        self, source, should_trip
    ):
        """Both halves matter. Without the True cases the audit could be
        vacuous; without the False cases it would ban the code it permits."""
        assert bool(audit_no_local_statistics(source)) is should_trip

    def test_the_page_prints_the_digits_analytics_produced(
        self, populated_run, populated_page
    ):
        """The reason the rule exists: a dashboard that computes its own win
        rate can disagree with the report, and then nobody can tell which is
        wrong. So the rendered text must be `analytics`' own."""
        measured = win_rate(populated_run.trades, split_label=SplitPhase.TRAIN)
        assert measured.is_defined
        expected = f"{measured.value:.6g} {measured.unit} (n={measured.n})"
        assert expected in populated_page.panel(PanelId.FLOW_SCORE_DISTRIBUTION).body

    def test_rates_are_printed_as_fractions_not_percentages(self, populated_page):
        """Converting a fraction to a percentage is a multiplication, and a
        100x slip in a dashboard is the error nobody audits."""
        body = populated_page.panel(PanelId.FLOW_SCORE_DISTRIBUTION).body
        table = body[body.index("<table>"):]
        assert "fraction" in table
        # Plotly's own CSS uses `width:100%`, so the assertion is scoped to the
        # table, which is where this module does its own formatting.
        assert "%" not in table


# ---------------------------------------------------------------------------
# A populated run, which exists only to exercise the drawing paths
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_run(make_trade) -> RunResult:
    """32 trades and a dipping equity curve, all labelled TRAIN.

    32 so one Flow Score bucket clears the committed `min_sample_n=30` and the
    win-rate overlay is actually computed; below it `analytics` suppresses the
    rate and there would be no number to compare against.
    """
    trades = tuple(
        make_trade(
            trade_id=f"t{index:02d}",
            rr=1.0,
            exit_price=18020.0 if index % 2 == 0 else 17980.0,
            signal_ts=T0 + timedelta(days=index),
            regime=Regime.LOW_VOL if index % 3 else Regime.HIGH_VOL,
            split_label=SplitPhase.TRAIN,
        )
        for index in range(32)
    )
    equity: list[EquityPoint] = []
    peak = START_EQUITY
    level = START_EQUITY
    for index in range(40):
        level = level + (200.0 if index % 2 == 0 else -150.0)
        peak = max(peak, level)
        equity.append(
            EquityPoint(
                ts=T0 + timedelta(days=index),
                equity=level,
                realized_equity=level,
                high_water_mark=peak,
            )
        )
    return RunResult(
        provenance=RunProvenance(
            run_id="run-populated",
            symbol="NQ",
            config_hash="cfg0",
            data_hash="data0",
            seed=20240101,
            split_label=SplitPhase.TRAIN,
            starting_equity=START_EQUITY,
            dataset={"source": "synthetic_v1", "synthetic": True},
        ),
        trades=trades,
        equity=tuple(equity),
    )


@pytest.fixture
def populated_page(populated_run) -> DashboardPage:
    return build_page(collect(populated_run))


class TestPopulatedRendering:
    def test_the_outcome_panels_are_defined_once_there_are_trades(
        self, populated_page
    ):
        for panel_id in (
            PanelId.EQUITY_CURVE,
            PanelId.DRAWDOWN_CURVE,
            PanelId.DAILY_PNL,
            PanelId.CUMULATIVE_PNL,
            PanelId.FLOW_SCORE_DISTRIBUTION,
            PanelId.REGIME_PERFORMANCE,
            PanelId.TRADE_DISTRIBUTION,
        ):
            panel = populated_page.panel(panel_id)
            assert panel.is_defined, panel_id
            assert panel.body

    def test_each_figure_has_a_stable_div_id(self, populated_page):
        """Plotly mints a random uuid when it is not given an id, which would
        make two renders of one run differ byte for byte."""
        for panel in populated_page.panels:
            if panel.is_defined and "plotly" in panel.body:
                assert f'id="{panel.panel_id.value}"' in panel.body

    def test_two_renders_of_a_populated_run_are_byte_identical(self, populated_run):
        assert build_html(populated_run, inline_plotlyjs=False) == build_html(
            populated_run, inline_plotlyjs=False
        )

    def test_the_cumulative_panel_says_it_did_not_apply_the_offset(
        self, populated_page
    ):
        note = populated_page.panel(PanelId.CUMULATIVE_PNL).note
        assert "less the starting balance" in note
        assert "no arithmetic" in note

    def test_the_regime_panel_keeps_regimes_with_no_trades(self, populated_page):
        """An unvisited regime is a fact about the sample, not a row to omit."""
        body = populated_page.panel(PanelId.REGIME_PERFORMANCE).body
        for regime in Regime:
            assert str(regime) in body

    def test_suppressed_cells_show_no_rate_at_all(self, populated_page):
        """`min_sample_n` suppression is analytics'; the page must render the
        suppression rather than fill the gap."""
        body = populated_page.panel(PanelId.REGIME_PERFORMANCE).body
        assert "insufficient_sample" in body or "no_trades" in body

    def test_a_run_whose_trades_disagree_with_its_label_is_refused(
        self, populated_run, make_trade
    ):
        """Section 0.8's refusal reaches the dashboard, because the dashboard
        asks analytics rather than counting for itself."""
        mislabelled = populated_run.replace(
            trades=(
                *populated_run.trades[:31],
                make_trade(trade_id="odd", split_label=SplitPhase.TEST),
            )
        )
        with pytest.raises(AnalyticsError):
            collect(mislabelled)


# ---------------------------------------------------------------------------
# Monte Carlo: an input that does not exist yet
# ---------------------------------------------------------------------------


def _small_plan(**changes) -> MonteCarloPlan:
    """A cheap plan. Nothing here is a config default: `MonteCarloPlan.default()`
    builds from the schema in-process and only the simulation count and the
    stored-curve count are reduced, neither of which can move a percentile's
    definition."""
    return MonteCarloPlan.default().replace(
        n_simulations=200, n_curves_to_store=4, **changes
    )


@pytest.fixture
def mc_refusal():
    """The live case: no trades, so the engine refuses rather than fanning."""
    return run_monte_carlo((), split_label=None, plan=_small_plan())


@pytest.fixture
def mc_distribution(populated_run):
    return run_monte_carlo(
        populated_run.trades, split_label=SplitPhase.TRAIN, plan=_small_plan()
    )


class TestMonteCarloPanels:
    def test_both_panels_say_the_input_was_not_supplied(self, real_page):
        for panel_id in (PanelId.MC_PERCENTILE_FAN, PanelId.MC_PERCENTILE_TABLE):
            panel = real_page.panel(panel_id)
            assert panel.undefined is Undefined.INPUT_NOT_SUPPLIED
            assert "No Monte Carlo result was supplied" in panel.note

    def test_a_refusal_renders_its_own_reason_not_a_flat_fan(
        self, real_empty_run, mc_refusal
    ):
        """The live path end to end: a run with no trades, the engine's refusal,
        and two panels that say NO_TRADES rather than drawing a flat envelope at
        the starting balance."""
        assert isinstance(mc_refusal, MonteCarloRefusal)
        assert mc_refusal.undefined is Undefined.NO_TRADES
        assert not hasattr(mc_refusal, "fan")
        page = build_page(collect(real_empty_run, monte_carlo=mc_refusal))
        for panel_id in (PanelId.MC_PERCENTILE_FAN, PanelId.MC_PERCENTILE_TABLE):
            panel = page.panel(panel_id)
            assert panel.undefined is Undefined.NO_TRADES, panel_id
            assert panel.body == ""
            assert "LIMITATION:" in panel.note

    def test_a_distribution_renders_the_fan_and_the_table(
        self, populated_run, mc_distribution
    ):
        page = build_page(collect(populated_run, monte_carlo=mc_distribution))
        fan = page.panel(PanelId.MC_PERCENTILE_FAN)
        assert fan.is_defined
        table = page.panel(PanelId.MC_PERCENTILE_TABLE)
        assert table.is_defined
        for label in ("p5", "p25", "p50", "p75", "p95"):
            assert label in table.body

    def test_the_fan_plots_exactly_the_engines_bands(
        self, populated_run, mc_distribution
    ):
        page = build_page(collect(populated_run, monte_carlo=mc_distribution))
        body = page.panel(PanelId.MC_PERCENTILE_FAN).body
        for band in mc_distribution.fan.bands:
            assert f"p{band.percentile:g}" in body

    def test_n_is_the_resample_pool_not_the_simulation_count(
        self, populated_run, mc_distribution
    ):
        """The engine's convention, carried onto the panel: 200 simulations over
        32 trades is 32 trades' worth of evidence, and labelling the panel 200
        would overstate the sample by a factor the reader cannot see."""
        page = build_page(collect(populated_run, monte_carlo=mc_distribution))
        assert mc_distribution.n_simulations != mc_distribution.n_trades
        for panel_id in (PanelId.MC_PERCENTILE_FAN, PanelId.MC_PERCENTILE_TABLE):
            assert page.panel(panel_id).n == mc_distribution.n_trades

    def test_a_percentile_the_plan_did_not_ask_for_stays_undefined(
        self, populated_run
    ):
        """Section 12 names five percentiles. A plan that computed two must not
        produce a table with three interpolated values in it."""
        thin = run_monte_carlo(
            populated_run.trades,
            split_label=SplitPhase.TRAIN,
            plan=_small_plan(percentiles=(5.0, 50.0)),
        )
        page = build_page(collect(populated_run, monte_carlo=thin))
        body = page.panel(PanelId.MC_PERCENTILE_TABLE).body
        assert body.count("input_not_supplied") == 3

    def test_the_union_does_not_coerce_a_refusal_into_a_distribution(
        self, real_empty_run, mc_refusal
    ):
        """`DashboardInputs.monte_carlo` is a union of two models sharing a base.
        If pydantic widened a refusal into the populated type, the n=0 panels
        would silently acquire fields the engine refuses to give them."""
        held = collect(real_empty_run, monte_carlo=mc_refusal).monte_carlo
        assert isinstance(held, MonteCarloRefusal)
        assert held.undefined is Undefined.NO_TRADES
        assert held == mc_refusal

    def test_both_panels_carry_the_limitation_verbatim(
        self, populated_run, mc_distribution
    ):
        """Read off the result rather than restated here: the engine validates
        `limitation` against STATED_LIMITATION exactly, so a paraphrase in the
        dashboard would be the one copy nothing checks."""
        page = build_page(collect(populated_run, monte_carlo=mc_distribution))
        for panel_id in (PanelId.MC_PERCENTILE_FAN, PanelId.MC_PERCENTILE_TABLE):
            note = page.panel(panel_id).note
            assert mc_distribution.limitation in note
            assert "SEQUENCE risk, not MODEL risk" in note


# ---------------------------------------------------------------------------
# Walk-forward and IS/OOS: tables, which render at n=0
# ---------------------------------------------------------------------------


class _Cfg:
    config_hash = "cfg0000"


class _EmptyRunner:
    """A runner that takes no trades in every window -- the live case."""

    def __call__(self, config, split) -> RunResult:
        return RunResult(
            provenance=RunProvenance(
                run_id=f"run-{split.name}",
                symbol="NQ",
                config_hash=_Cfg.config_hash,
                data_hash="data0",
                seed=20240101,
                split_label=split.phase,
                split_name=split.name,
            )
        )


@pytest.fixture
def walk_forward_report(tmp_path):
    guard = DataAccessGuard(
        sealed=None, audit=SealAudit(tmp_path / "audit.jsonl"), enabled=False
    )
    folds = generate_folds(
        DateRange(start=date(2016, 1, 1), end=date(2016, 4, 1)),
        WindowSpec(
            train_days=30,
            validation_days=10,
            test_days=10,
            step_days=10,
            purge_days=1,
            embargo_days=2,
            min_trades_per_fold=0,
        ),
        guard=guard,
    )
    return evaluate_folds(folds, _Cfg(), _EmptyRunner())


class TestWalkForwardAndSplitPanels:
    def test_the_fold_table_lists_every_fold_even_at_n_zero(
        self, real_empty_run, walk_forward_report
    ):
        """"Do not hide bad periods" starts with the fold that has no trades.
        A table of undefined cells is a report; a dropped row is a filter."""
        page = build_page(collect(real_empty_run, walk_forward=walk_forward_report))
        panel = page.panel(PanelId.WALK_FORWARD_FOLDS)
        assert panel.is_defined
        assert panel.n == walk_forward_report.n_folds
        for result in walk_forward_report.folds:
            assert f"{result.fold.index:03d}" in panel.body
        assert "no_trades" in panel.body

    def test_the_fold_panel_names_no_best_fold(self, real_empty_run, walk_forward_report):
        page = build_page(collect(real_empty_run, walk_forward=walk_forward_report))
        assert "best" in page.panel(PanelId.WALK_FORWARD_FOLDS).note

    def test_a_missing_report_is_an_absent_input_not_an_empty_study(self, real_page):
        panel = real_page.panel(PanelId.WALK_FORWARD_FOLDS)
        assert panel.undefined is Undefined.INPUT_NOT_SUPPLIED

    def test_one_phase_is_not_a_comparison(self, real_page):
        panel = real_page.panel(PanelId.IS_VS_OOS)
        assert panel.undefined is Undefined.INPUT_NOT_SUPPLIED
        assert "only one split phase" in panel.note.lower()

    def test_two_phases_render_as_two_labelled_columns_and_no_combined_one(
        self, populated_run, make_trade
    ):
        oos = performance_summary(
            (make_trade(trade_id="oos1", split_label=SplitPhase.TEST),),
            split_label=SplitPhase.TEST,
        )
        page = build_page(collect(populated_run, extra_summaries=(oos,)))
        panel = page.panel(PanelId.IS_VS_OOS)
        assert panel.is_defined
        assert "TRAIN" in panel.body
        assert "TEST" in panel.body
        assert "in-sample (fitting)" in panel.body
        assert "out-of-sample" in panel.body
        for forbidden in ("combined", "overall", "all phases"):
            assert forbidden not in panel.body.lower()

    def test_two_summaries_under_one_label_are_refused(self, populated_run):
        duplicate = performance_summary(
            populated_run.trades, split_label=SplitPhase.TRAIN
        )
        with pytest.raises(ValidationError, match="same split label"):
            collect(populated_run, extra_summaries=(duplicate,))

    def test_a_summary_filed_under_the_wrong_phase_is_refused(self, populated_run):
        wrong = performance_summary((), split_label=SplitPhase.TEST)
        with pytest.raises(ValidationError, match="wrong phase"):
            DashboardInputs(
                run=populated_run,
                summaries=(wrong,),
                flow_score=collect(populated_run).flow_score,
                regime=collect(populated_run).regime,
            )


class TestTheWholePage:
    def test_a_fully_supplied_page_renders_and_passes_its_own_origin_check(
        self, populated_run, mc_distribution, walk_forward_report, make_trade
    ):
        """Everything the brief names, supplied at once: a run, a second split
        phase, a Monte Carlo distribution and a walk-forward report."""
        oos = performance_summary(
            (make_trade(trade_id="oos1", split_label=SplitPhase.TEST),),
            split_label=SplitPhase.TEST,
        )
        page = build_page(
            collect(
                populated_run,
                monte_carlo=mc_distribution,
                walk_forward=walk_forward_report,
                extra_summaries=(oos,),
            )
        )
        assert page.n_undefined == 0
        document = render(page, inline_plotlyjs=False)
        assert_origin_is_stated(document, page)
        for panel in page.panels:
            assert panel.provenance.strip()
            assert "synthetic=True" in panel.provenance, panel.panel_id

    def test_the_fully_supplied_page_is_deterministic(
        self, populated_run, mc_distribution, walk_forward_report
    ):
        def once() -> str:
            return render(
                build_page(
                    collect(
                        populated_run,
                        monte_carlo=mc_distribution,
                        walk_forward=walk_forward_report,
                    )
                ),
                inline_plotlyjs=False,
            )

        assert once() == once()
