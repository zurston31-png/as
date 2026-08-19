"""The single document that answers "does this strategy work?".

Every other module in app/analysis/ and app/research/ measures one thing.
This assembles them into one verdict per question, using three states and
only three:

    PASS               measured, and the answer is yes
    FAIL               measured, and the answer is no
    INSUFFICIENT DATA  not measured, or measured on too little to mean
                       anything

The third state does most of the work. Almost everything about a young
strategy is INSUFFICIENT DATA, and saying so is the honest output. A report
that resolved every question into PASS or FAIL would be inventing
confidence, and the specific confidence it invents is always "the strategy
looks fine", because a small sample of a random strategy usually does.

FAIL IS A SUCCESSFUL OUTCOME. Learning that a strategy has no edge is worth
more than learning it might: it stops you spending months tuning something
that cannot work. Nothing in here is written to find a way to say yes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session

from app import models
from app.analysis import forward_returns as fr
from app.analysis import trade_analytics as ta
from app.analysis.calibration import HORIZONS_MINUTES, build_calibration
from app.analysis.report import build_performance_report
from app.analysis.score_distribution import build_score_distribution
from app.analysis.stage_funnel import build_stage_funnel
from app.analysis.validation import ValidationStatus
from app.safety import reconcile as reconcile_mod


class Grade(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT DATA"


@dataclass
class Finding:
    question: str
    grade: Grade
    detail: str

    def as_dict(self) -> dict:
        return {"question": self.question, "grade": self.grade.value, "detail": self.detail}


@dataclass
class ResearchReport:
    generated_at: dt.datetime
    findings: list[Finding] = field(default_factory=list)
    headline: str = ""
    weaknesses: list[str] = field(default_factory=list)
    data_needed: list[str] = field(default_factory=list)

    def grade(self, question: str) -> Grade | None:
        found = next((f for f in self.findings if f.question == question), None)
        return found.grade if found else None

    @property
    def counts(self) -> dict[str, int]:
        return {
            g.value: sum(1 for f in self.findings if f.grade is g) for g in Grade
        }

    @property
    def any_failures(self) -> bool:
        return any(f.grade is Grade.FAIL for f in self.findings)

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "headline": self.headline,
            "counts": self.counts,
            "findings": [f.as_dict() for f in self.findings],
            "weaknesses": list(self.weaknesses),
            "data_needed": list(self.data_needed),
        }

    def render(self) -> str:
        width = 78
        lines = [
            "=" * width,
            " STRATEGY RESEARCH REPORT",
            "=" * width,
            f" {self.headline}",
            "",
            f" PASS {self.counts['PASS']}   FAIL {self.counts['FAIL']}   "
            f"INSUFFICIENT DATA {self.counts['INSUFFICIENT DATA']}",
            "",
        ]
        for f in self.findings:
            lines.append(f"  [{f.grade.value:^17}]  {f.question}")
            lines.append(f"                       {f.detail}")
            lines.append("")
        if self.weaknesses:
            lines += ["-" * width, " STRONGEST REMAINING WEAKNESSES", "-" * width]
            lines += [f"  {i}. {w}" for i, w in enumerate(self.weaknesses, 1)]
            lines.append("")
        if self.data_needed:
            lines += ["-" * width, " WHAT WOULD MAKE THIS ANSWERABLE", "-" * width]
            lines += [f"  - {d}" for d in self.data_needed]
            lines.append("")
        lines += [
            "=" * width,
            " Paper trading only. No real funds, wallet or private key is involved.",
            "=" * width,
        ]
        return "\n".join(lines)


def _expectancy_finding(report) -> Finding:
    stats = report.stats
    if stats.trade_count == 0:
        return Finding(
            "Does the strategy show positive expectancy after costs?",
            Grade.INSUFFICIENT,
            "No closed trades yet.",
        )
    if stats.trade_count < 100:
        return Finding(
            "Does the strategy show positive expectancy after costs?",
            Grade.INSUFFICIENT,
            f"Expectancy is ${stats.expectancy_usd:+,.2f} per trade over {stats.trade_count} "
            "trades - too few for that number to mean anything. It is arithmetic, not evidence.",
        )
    if stats.expectancy_usd > 0:
        return Finding(
            "Does the strategy show positive expectancy after costs?",
            Grade.PASS,
            f"${stats.expectancy_usd:+,.2f} per trade over {stats.trade_count} closed trades, "
            f"net of ${report.costs.total_execution_cost_usd:,.2f} in recorded execution costs.",
        )
    return Finding(
        "Does the strategy show positive expectancy after costs?",
        Grade.FAIL,
        f"${stats.expectancy_usd:+,.2f} per trade over {stats.trade_count} trades. "
        "The strategy loses money after costs.",
    )


def _sample_finding(report) -> Finding:
    n = report.stats.trade_count
    if n >= 100:
        return Finding("How large is the sample?", Grade.PASS,
                       f"{n} closed trades - enough for the headline statistics to be read.")
    return Finding(
        "How large is the sample?", Grade.INSUFFICIENT,
        f"{n} closed trades. Below ~100 a win rate's confidence interval spans "
        "'excellent' and 'losing' simultaneously.",
    )


def _cost_finding(report) -> Finding:
    costs = report.costs
    if costs.legs_counted == 0:
        return Finding("Does performance survive realistic costs?", Grade.INSUFFICIENT,
                       "No execution costs have been recorded yet.")
    if not costs.cost_data_complete:
        return Finding(
            "Does performance survive realistic costs?", Grade.INSUFFICIENT,
            f"Only {costs.coverage_pct:.0f}% of filled legs have recorded costs, so the "
            "cost total is understated and any 'after costs' figure is optimistic.",
        )
    gross, net = report.gross_pnl_usd, report.net_pnl_usd
    if net > 0:
        return Finding(
            "Does performance survive realistic costs?", Grade.PASS,
            f"${gross:,.2f} gross becomes ${net:,.2f} net after "
            f"${costs.total_execution_cost_usd:,.2f} of fees, spread and impact.",
        )
    if gross > 0 >= net:
        return Finding(
            "Does performance survive realistic costs?", Grade.FAIL,
            f"Gross P&L is positive (${gross:,.2f}) but costs of "
            f"${costs.total_execution_cost_usd:,.2f} turn it negative (${net:,.2f}). "
            "The edge is smaller than the cost of trading.",
        )
    return Finding("Does performance survive realistic costs?", Grade.FAIL,
                   f"Negative before costs (${gross:,.2f}) and after (${net:,.2f}).")


def _calibration_finding(db: Session) -> Finding:
    question = "Does the score predict future outcomes?"
    coverage = fr.coverage(db)
    if coverage["resolved"] == 0:
        return Finding(
            question, Grade.INSUFFICIENT,
            "No forward returns resolved yet. The bot records what every scored candidate "
            "did afterwards, including rejected ones; that data has not accumulated.",
        )

    verdicts = []
    for horizon in HORIZONS_MINUTES:
        table = build_calibration(db, horizon_minutes=horizon)
        if table.monotonic is not None:
            verdicts.append((horizon, table))

    if not verdicts:
        return Finding(
            question, Grade.INSUFFICIENT,
            f"{coverage['resolved']} forward returns resolved ({coverage['coverage_pct']}% "
            "coverage) but no horizon has two score buckets with enough outcomes to compare.",
        )

    positive = [(h, t) for h, t in verdicts if t.monotonic]
    if positive:
        horizon, table = positive[0]
        return Finding(question, Grade.PASS, table.verdict())
    horizon, table = verdicts[0]
    return Finding(question, Grade.FAIL, table.verdict())


def _distribution_finding(db: Session) -> Finding:
    question = "What does the scoring engine actually produce?"
    dist = build_score_distribution(db)
    if not dist.sample_size:
        return Finding(question, Grade.INSUFFICIENT, "The engine has not scored anything yet.")
    if not dist.reliable:
        return Finding(
            question, Grade.INSUFFICIENT,
            f"Only {dist.sample_size} scores recorded. "
            f"Mean {dist.mean:.1f}, median {dist.median:.1f} - arithmetic, not a description.",
        )
    reaching_65 = dist.share_reaching(65)
    reaching_75 = dist.share_reaching(75)
    return Finding(
        question, Grade.PASS,
        f"n={dist.sample_size}. Mean {dist.mean:.1f}, median {dist.median:.1f}, "
        f"sd {dist.stdev:.1f}, range {dist.minimum:.1f}-{dist.maximum:.1f}. "
        f"{reaching_65 * 100:.0f}% reach 65, {reaching_75 * 100:.0f}% reach 75.",
    )


def _funnel_finding(db: Session) -> Finding:
    question = "Which filters eliminate the most candidates?"
    funnel = build_stage_funnel(db, window_hours=None)
    if not funnel.discovered:
        return Finding(question, Grade.INSUFFICIENT,
                       "No tokens discovered yet - the scanner has not run or found nothing.")
    return Finding(question, Grade.PASS, funnel.explain())


def _concentration_finding(report) -> Finding:
    question = "Is the strategy overly dependent on a few winners?"
    extremes = report.extremes
    if extremes.best_trade_share_of_profit is None:
        return Finding(question, Grade.INSUFFICIENT, "No profitable trades yet.")
    if report.stats.win_count < 10:
        return Finding(
            question, Grade.INSUFFICIENT,
            f"Only {report.stats.win_count} winning trade(s). With this few, one dominating "
            "the profit is arithmetic rather than concentration.",
        )
    share = extremes.best_trade_share_of_profit
    if share > 0.40:
        return Finding(
            question, Grade.FAIL,
            f"The single best trade is {share:.0%} of gross profit. The result rests on it.",
        )
    return Finding(question, Grade.PASS,
                   f"The best trade is {share:.0%} of gross profit - the edge is spread.")


def _drawdown_finding(report) -> Finding:
    question = "How severe are realistic drawdowns?"
    mc = report.monte_carlo
    if mc is None or not mc.reliable:
        return Finding(
            question, Grade.INSUFFICIENT,
            "Too few closed trades to resample. The realized drawdown of one path is not "
            "the drawdown to plan around.",
        )
    if mc.p95_max_drawdown_pct > 35:
        return Finding(
            question, Grade.FAIL,
            f"Resampling the same trades in different orders gives a 95th-percentile drawdown "
            f"of {mc.p95_max_drawdown_pct:.1f}% (worst {mc.worst_max_drawdown_pct:.1f}%), "
            f"against {report.stats.max_drawdown_pct:.1f}% realized.",
        )
    return Finding(
        question, Grade.PASS,
        f"95th-percentile resampled drawdown {mc.p95_max_drawdown_pct:.1f}% "
        f"(realized {report.stats.max_drawdown_pct:.1f}%).",
    )


def _integrity_finding(db: Session) -> Finding:
    question = "Is the recorded data internally consistent?"
    result = reconcile_mod.reconcile(db)
    problems = reconcile_mod.check_position_integrity(db)
    if not result.balanced or problems:
        detail = result.summary().splitlines()[0]
        if problems:
            detail += f" Also {len(problems)} position-book problem(s)."
        return Finding(question, Grade.FAIL, detail)
    return Finding(question, Grade.PASS,
                   f"Ledger reconciles across {result.filled_trades} filled trades; "
                   "the open book is structurally sound.")


def _untested(question: str, what: str, how: str) -> Finding:
    return Finding(question, Grade.INSUFFICIENT, f"{what} {how}")


def build_research_report(db: Session, *, strategy_version: str | None = None) -> ResearchReport:
    """Answer every validation question from whatever data exists."""
    performance = build_performance_report(db, strategy_version=strategy_version,
                                           monte_carlo_simulations=2_000)

    findings = [
        _sample_finding(performance),
        _expectancy_finding(performance),
        _cost_finding(performance),
        _distribution_finding(db),
        _calibration_finding(db),
        _funnel_finding(db),
        _concentration_finding(performance),
        _drawdown_finding(performance),
        _integrity_finding(db),
        _untested(
            "Does it survive out-of-sample testing?",
            "Not run against live-recorded data.",
            "Run scripts/run_backtest.py over recorded history to answer this.",
        ),
        _untested(
            "Does it survive walk-forward testing?",
            "Not run.",
            "Run scripts/research.py walk-forward once enough history exists.",
        ),
        _untested(
            "Does performance persist across market regimes?",
            "Not measured.",
            "Needs recorded history spanning more than one regime.",
        ),
        _untested(
            "Which indicators help and which hurt?",
            "Ablation not run.",
            "Run scripts/research.py ablate - it needs enough history for 20+ "
            "out-of-sample trades per variant.",
        ),
        _untested(
            "How stable are the parameters?",
            "Robustness sweep not run.",
            "Run scripts/research.py sweep / thresholds.",
        ),
    ]

    counts = {g.value: sum(1 for f in findings if f.grade is g) for g in Grade}
    validation = performance.validation

    if counts["FAIL"]:
        headline = (
            f"{counts['FAIL']} question(s) answered NO on the evidence available. "
            "The strategy is NOT validated, and some of what is known is negative."
        )
    elif validation and validation.status is ValidationStatus.VALIDATED:
        headline = (
            "Every measured question passed. This is evidence about the STRATEGY, not "
            "about real execution - every fill in this record was simulated."
        )
    else:
        headline = (
            f"NOT VALIDATED. {counts['INSUFFICIENT DATA']} of {len(findings)} questions cannot "
            "be answered yet. Nothing here says the strategy works, and nothing here says it "
            "does not."
        )

    weaknesses = []
    if performance.stats.trade_count < 100:
        weaknesses.append(
            f"Sample size: {performance.stats.trade_count} closed trades. Every performance "
            "figure below ~100 is dominated by luck."
        )
    if not performance.costs.cost_data_complete and performance.costs.legs_counted:
        weaknesses.append(
            f"Execution-cost coverage is {performance.costs.coverage_pct:.0f}%, so reported "
            "costs are understated and net figures are optimistic."
        )
    coverage = fr.coverage(db)
    if coverage["total"] and coverage["coverage_pct"] < 60:
        weaknesses.append(
            f"Forward-return coverage is {coverage['coverage_pct']}%. Calibration built on this "
            "describes whichever tokens stayed liquid, which is the bias it exists to avoid."
        )
    if len(performance.version_counts) > 1:
        weaknesses.append(
            f"The record spans {len(performance.version_counts)} strategy versions. Pooled "
            "figures describe no single strategy."
        )
    if not weaknesses:
        weaknesses.append(
            "No structural weakness found in the data itself - but see the INSUFFICIENT DATA "
            "findings above, which are gaps in evidence rather than in the data's quality."
        )

    data_needed = [
        f"{max(0, 100 - performance.stats.trade_count)} more closed paper trades to reach a "
        "readable sample.",
        "Resolved forward returns across several score buckets, to answer whether the score "
        "predicts anything. This accumulates on its own while the bot runs.",
        "Recorded candle history long enough for a 60/40 train/out-of-sample split to leave "
        "20+ trades on the out-of-sample side, which is what ablation and threshold research "
        "need before they can say anything.",
        "History spanning more than one market regime, so regime-conditional performance can "
        "be separated from a single lucky period.",
    ]

    return ResearchReport(
        generated_at=dt.datetime.now(dt.timezone.utc),
        findings=findings,
        headline=headline,
        weaknesses=weaknesses,
        data_needed=data_needed,
    )
