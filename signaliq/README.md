# SignalIQ - Milestone 1: validated historical market data

SignalIQ is a validation-first stock market data platform. This first milestone is the
foundation: it downloads real historical daily bars for ten US symbols from Alpaca's
official Market Data API, checks every bar, stores the good ones in a local SQLite
database and serves them through a small read-only web API.

**What it is not.** There are no trading strategies, predictions, buy/sell
recommendations, backtests or automated trading here. The ten symbols
(AAPL, MSFT, NVDA, AMD, AMZN, META, TSLA, GOOGL, SPY, QQQ) are a starter list and do
not represent the whole US stock market.

## Contents

- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [API reference](#api-reference)
- [Data coverage](#data-coverage)
- [How the data is handled](#how-the-data-is-handled)
- [Real data and test fixtures](#real-data-and-test-fixtures)
- [Database](#database)
- [Configuration](#configuration)
- [Tests](#tests)
- [Verification status](#verification-status)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Next steps: Milestone 2](#next-steps-milestone-2)

## Quick start

You need **Python 3.12** and a free **Alpaca** account. All commands are run from this
`signaliq` folder.

### 1. Create a virtual environment and install the dependencies

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keep the virtual environment activated for every command below. You will see `(.venv)`
at the start of the prompt.

### 2. Add your Alpaca API keys

1. Sign in at <https://app.alpaca.markets> and create API keys. Paper-trading keys are
   enough for market data.
2. Copy the example file, then open `.env` in a text editor and replace the two
   placeholder values with your own keys.

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

```ini
APCA_API_KEY_ID=your-key-id
APCA_API_SECRET_KEY=your-secret-key
```

The `.env` file stays on your machine. It is listed in `.gitignore`, and the keys are
never written to logs.

#### Managed runtimes where a proxy holds the secret key

Some runtimes keep the secret key out of the process entirely: an authenticating network
proxy adds the `APCA-API-SECRET-KEY` header to requests for `data.alpaca.markets`. A
Claude Code cloud environment does this when the secret is saved as a network secret for
that host. There, set only the key ID and switch the option on:

```ini
APCA_API_KEY_ID=your-key-id
SIGNALIQ_ALPACA_SECRET_FROM_PROXY=true
```

SignalIQ then sends the key ID header itself and never holds the secret. If
`APCA_API_SECRET_KEY` is also set, it is sent as usual. Only the market data host is
involved; SignalIQ has no trading code and needs no trading host.

In a Claude Code cloud environment both values are saved in the environment's settings
(the environment menu, then Edit): the secret key as a network secret for
`data.alpaca.markets` with header `APCA-API-SECRET-KEY`, and `APCA_API_KEY_ID` and
`SIGNALIQ_ALPACA_SECRET_FROM_PROXY=true` as environment variables. No trading host needs
to be on the network allowlist.

### 3. Create the database

```bash
python -m app.database
```

This creates `data/signaliq.db` with three empty tables. It is safe to run again.

### 4. Download historical bars

One ticker:

```bash
python scripts/ingest.py --symbol AAPL
```

All ten tickers:

```bash
python scripts/ingest.py --all
```

By default this requests daily bars from 2016-01-01 up to the last completed trading
session. A summary table is printed at the end. Running the same command again is safe:
bars that are already stored are skipped. All ten symbols take a few seconds.

Alpaca's IEX feed does not go back to 2016, so the summary reports the sessions before
its first bar as a warning. See [Data coverage](#data-coverage) for what really arrives.

### 5. Validate what is stored

```bash
python scripts/validate.py
```

### 6. Look at the database

```bash
python scripts/inspect_db.py
```

### 7. Start the API

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> in a browser for interactive documentation, or try:

```bash
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/symbols"
curl "http://127.0.0.1:8000/bars/AAPL?start=2024-01-02&end=2024-01-31"
curl "http://127.0.0.1:8000/ingestion/runs"
curl "http://127.0.0.1:8000/quality/report"
```

Stop the server with `Ctrl+C`.

### 8. Run the tests

```bash
pytest
```

## Command reference

| Purpose | Command |
| --- | --- |
| Initialise the database | `python -m app.database` |
| Download one ticker | `python scripts/ingest.py --symbol AAPL` |
| Download several tickers | `python scripts/ingest.py --symbol AAPL --symbol MSFT` or `--symbol AAPL,MSFT` |
| Download all ten tickers | `python scripts/ingest.py --all` |
| Choose the date range | add `--start 2020-01-01 --end 2020-12-31` |
| Validate stored bars | `python scripts/validate.py` (add `--symbol AAPL` for one symbol) |
| Inspect the database | `python scripts/inspect_db.py` (add `--json` for JSON output). Shows row counts, the date range and gap-free start of every symbol, the ingestion history and the check outcomes |
| Start the API | `uvicorn app.main:app --reload` |
| Run the tests | `pytest` |
| Help for any script | add `--help` |

Exit codes: `ingest.py` returns 0 when every run finished, 1 when a run failed and 2 for
an invalid command or configuration. `validate.py` returns 1 when any check failed, so it
can be used in automation.

## API reference

The API is read-only. It reads the local database and never calls Alpaca.

| Endpoint | What it returns |
| --- | --- |
| `GET /health` | Service status, database status, version and current UTC time |
| `GET /symbols` | The configured symbols with the stored bar count, date range and session coverage of each |
| `GET /bars/{symbol}` | Stored bars for one symbol |
| `GET /ingestion/runs` | Download history, newest first |
| `GET /quality/report` | Data-quality outcomes and the most recent issues |

`GET /bars/{symbol}` accepts these query parameters:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `timeframe` | Bar timeframe, for example `1Day` | `1Day` |
| `start` | First date or date-time to include | none |
| `end` | Last date or date-time to include | none |
| `limit` | Maximum bars to return (1 to 10000) | 1000 |
| `offset` | Bars to skip, for paging | 0 |
| `order` | `asc` (oldest first) or `desc` | `asc` |

A plain date such as `2024-01-05` is read as a New York market date, and both `start` and
`end` are inclusive. A date-time must carry a UTC offset, for example
`2024-01-05T14:30:00Z`. All timestamps in responses are UTC.

For daily bars, each entry of `GET /symbols` carries a `session_coverage` object that
compares what is stored with the trading calendar (New York dates):

| Field | Meaning |
| --- | --- |
| `first_session`, `last_session` | First and last trading session with a bar |
| `sessions_expected` | Trading sessions between the two, per the calendar |
| `sessions_missing` | How many of those have no bar |
| `gap_free_from` | From this session to `last_session`, every expected session has a bar |
| `gap_free_sessions` | Number of sessions in that uninterrupted run |

Anything that needs an unbroken daily series should read from `gap_free_from`. Bars
stored before it are still returned by `GET /bars/{symbol}`; nothing is hidden or filled.

`GET /ingestion/runs` accepts `symbol`, `status`, `limit` and `offset`.
`GET /quality/report` accepts `symbol`, `stage` (`ingestion` or `stored`) and `limit`.

Invalid input returns a clear message: an unknown symbol returns 404 with the list of
supported symbols, and a malformed date, timeframe or paging value returns 422.

## Data coverage

What the full download really contains. Measured on 2026-11-17 (UTC) with
`python scripts/ingest.py --all`, feed `iex`, adjustment `split`, timeframe `1Day`,
requested from 2016-01-01 to the last completed session (2026-11-16).

| Symbol | Bars stored | First bar | Gap-free from | Last bar | Missing sessions after the gap-free start |
| --- | --- | --- | --- | --- | --- |
| AAPL | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| MSFT | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| NVDA | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| AMD | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| AMZN | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| META | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| TSLA | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| GOOGL | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |
| SPY | 1,587 | 2018-11-01 | 2020-07-27 | 2026-11-16 | 0 |
| QQQ | 1,586 | 2020-07-27 | 2020-07-27 | 2026-11-16 | 0 |

15,861 bars in total. From 2020-07-27 to 2026-11-16 the New York Stock Exchange held
1,586 sessions, and every symbol has a bar for every one of them. Re-run
`python scripts/inspect_db.py` to see the same table for your own database; the dates
move forward as new sessions complete.

### What the provider does not have

- **Alpaca's IEX daily history starts on 2020-07-27.** Requests for 2016-01-01 to
  2020-07-26 return no bars for any of the ten symbols (checked directly against the
  API). The 1,148 expected sessions from 2016-01-04 to 2020-07-24 are therefore reported
  as a `trading_session_gap` **warning** for each symbol. Nothing was filled in.
- **SPY has one isolated bar on 2018-11-01**: a single trade of 200 shares, open = high =
  low = close = 271.60. The provider serves it and then nothing until 2020-07-27. It is
  stored exactly as received. Because 433 expected sessions (2018-11-02 to 2020-07-24)
  lie between it and the rest of the series, validation reports one
  `trading_session_gap` **failure** for SPY, SPY's ingestion runs end as
  `completed_with_issues`, and `validate.py` exits with code 1. This is deliberate: the
  finding is real and is not suppressed. Use `gap_free_from` (2020-07-27) to read SPY
  without the stray bar.
- **IEX is one exchange, not the whole market.** Volume and trade counts are IEX-only
  and far below consolidated market volume; open, high, low and close are IEX prices and
  can differ slightly from the official consolidated values. Do not read IEX volume as
  market volume. On an early-close day, or for a thinly traded symbol, an IEX daily bar
  can rest on few trades.
- **Prices are split-adjusted by the provider.** The splits inside the stored range
  (AAPL and TSLA in 2020, NVDA in 2021 and 2024, AMZN, GOOGL and TSLA in 2022) do not
  appear as jumps: the largest day-to-day close change of any symbol is 26.3 %
  (META, 2022-02-03). Dividends are not adjusted for.
- **META before June 2022.** The provider returns the history of the former ticker FB
  under META: the META series is unbroken from 2020-07-27 across the rename.

### The SIP feed (not used)

Read-only probes on 2026-11-17, with the same account, showed that Alpaca's consolidated
`sip` feed does have the older history: AAPL and SPY each returned 1,148 daily bars for
2016-01-04 to 2020-07-24, which are exactly the sessions the IEX feed lacks. A `sip`
request is refused with HTTP 403 ("subscription does not permit querying recent SIP
data") when its end time is less than about 15 minutes in the past; with an end 20
minutes back it succeeded up to the latest session.

None of this was stored. Milestone 1 is specified for the IEX feed, SIP prices and
volumes are consolidated values that differ from IEX ones, and a bar is identified by
symbol, timeframe and timestamp only, so the two feeds cannot yet be kept side by side.
Whether to add SIP is listed under [Next steps](#next-steps-milestone-2).

## How the data is handled

**Source.** Bars come from `GET https://data.alpaca.markets/v2/stocks/bars`, authenticated
with the `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY` headers.

**Feed.** Every request asks for `feed=iex` explicitly. If Alpaca refuses the feed, the
download fails with a clear error. SignalIQ never switches to another feed on its own.
IEX is a single exchange, so volumes and trade counts are IEX-only and are much smaller
than total market volume, and prices can differ slightly from the consolidated tape.

**Adjustment and currency.** Requests use `adjustment=split` and `currency=USD`.

**Completed candles only.** The request end date is limited to the last completed trading
session. A session counts as complete from 20:15 New York time on that day, after
extended-hours trading has ended. Today's candle is never requested while it can still
change.

**Pagination.** The client follows `next_page_token` until Alpaca reports no further
page.

**Reliability.** Requests have a timeout. Timeouts, connection errors and 5xx responses
are retried with exponential backoff. HTTP 429 waits for the time Alpaca indicates
before retrying. Credential and permission errors (401, 403) are not retried.

**Timestamps.** Everything is stored in UTC as text ending in `Z`. A timestamp without a
timezone is refused. Alpaca stamps a daily bar at midnight New York time, which is
`05:00Z` in winter and `04:00Z` in summer.

**Idempotent imports.** A bar is identified by symbol, timeframe and timestamp. On
re-import, a new bar is inserted and an identical bar is skipped.

**Conflicts are recorded, not overwritten.** If a re-downloaded bar differs from the
stored one, the stored bar is kept and the difference is written to
`data_quality_checks` for investigation.

**Invalid bars are rejected, not repaired.** A bar that fails a check is not stored. It is
recorded with its original payload. Missing prices are never filled in and candles are
never invented.

### Validation checks

| Check name | What it catches |
| --- | --- |
| `missing_required_fields` | A required field is missing, null or not a number |
| `non_positive_price` | Open, high, low or close is zero or negative |
| `high_below_low` | High is lower than low |
| `open_close_outside_range` | Open or close is outside the low-high range |
| `negative_volume` | Volume is negative |
| `duplicate_timestamp` | The same timestamp appears more than once for a symbol |
| `invalid_timestamp` | Timestamp is malformed, has no timezone or is not UTC |
| `unexpected_symbol` | Symbol was not requested or is not in the configured list |
| `invalid_source_metadata` | Source, feed or adjustment is missing or not as configured |
| `trading_session_gap` | An expected trading session has no bar |
| `conflicting_record` | A downloaded bar differs from the stored bar |

Each check outcome has a status: `passed`, `failed`, `warning` or `skipped`.

### Missing sessions and the trading calendar

Gap detection compares stored bars with a built-in New York Stock Exchange calendar, so
weekends and market holidays are never reported as missing. The calendar covers the ten
regular holidays (including Good Friday, and Juneteenth from 2022) and the unscheduled
closures of 2001, 2004, 2007, 2012, 2018 and 2025. It was compared day by day with the
independent `exchange_calendars` library for 2000 through 2027 and matched on all 7,041
sessions. Against the real data it also holds: the stored bars of all ten symbols fall
on exactly the 1,586 sessions that library lists from 2020-07-27 to 2026-11-16, with no
bar on a closed day.

**Early closes.** On the day after Thanksgiving, on Christmas Eve and on 3 July the
regular session can end at 13:00 New York time. Such a day is still a trading session:
a daily bar is expected, and a missing one is reported as a gap. The calendar knows
these days (`market_early_closes`, `session_close_time` in `app/data/validation.py`);
its rules match `exchange_calendars` for every year from 2004 through 2027. The twelve
early closes between November 2020 and December 2025 were also confirmed with real
hourly IEX bars (SPY volume after 13:00 was under 3 % of the morning's), and each of
them has a daily bar for all ten symbols. Daily ingestion does not depend on the closing
time: a candle is requested only after 20:15 New York time, which is later than the end
of trading on any day. Before 2004 the exchanges used one-off schedules that the rules
do not reproduce.

- A missing session between stored bars is a **failure**.
- Sessions missing before the first stored bar are a **warning**, because the provider's
  history for a feed may start later than 2016-01-01.
- Sessions missing after the last stored bar are a **warning** that the data needs
  updating.

## Real data and test fixtures

Two kinds of bars exist in this project and they never mix.

| | Real market data | Synthetic test fixtures |
| --- | --- | --- |
| Where | `data/signaliq.db`, written only by `scripts/ingest.py` | Throw-away databases created by `pytest` in a temporary folder |
| Symbols | The ten configured tickers | The fake tickers `TESTA` and `TESTB` |
| Values | Exactly what Alpaca's API returned | Invented numbers (open 10.0, high 11.0, low 9.0, close 10.5, volume 1,000) |
| Source | `source=alpaca`, `feed=iex`, tied to an `ingestion_runs` row | Built in `tests/conftest.py`; the HTTP responses are mocked |

The offline tests never touch `data/signaliq.db` and never call Alpaca. The fake tickers
cannot be stored by ingestion, because only configured symbols are accepted. The live
tests (`pytest -m integration`) do fetch real bars, but store them only in a temporary
database. They check structure and behaviour and assert nothing about price levels.
Nothing in this repository is a statement about market performance.

## Database

SQLite file: `data/signaliq.db`. It is an ordinary file, so the data survives restarts of
the API and of the machine. The file is listed in `.gitignore` and is not part of the
repository: on a new machine, or in a new cloud session, recreate it with
`python scripts/ingest.py --all`.

| Table | Contents |
| --- | --- |
| `bars` | One row per bar: symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count, source, feed, adjustment. Primary key: symbol, timeframe, timestamp. Each row also records the ingestion run that stored it and when. |
| `ingestion_runs` | One row per symbol and download attempt: requested window, feed, status, counters for fetched, inserted, unchanged, conflicting, rejected and duplicate bars, missing sessions, and any error message. |
| `data_quality_checks` | Check outcomes from ingestion and from `validate.py`. A `passed` row summarises one check; a `failed` or `warning` row describes one finding with the offending values. |

Uniqueness and indexes: the primary key of `bars` (symbol, timeframe, timestamp) is a
unique index, so the database itself refuses a second bar for the same key. Further
indexes exist on `bars.ingestion_run_id`, on `ingestion_runs` (symbol, started_at) and on
`data_quality_checks` (check_run_id, ingestion_run_id, status, and symbol with
check_name). Foreign keys are enforced.

## Configuration

Settings are read from `.env` and from environment variables. Only the credentials are
required: both keys, or the key ID together with `SIGNALIQ_ALPACA_SECRET_FROM_PROXY`
(see [Managed runtimes where a proxy holds the secret key](#managed-runtimes-where-a-proxy-holds-the-secret-key)).
The credentials are used for market data only. SignalIQ contains no trading code and
never contacts a trading host.

| Variable | Meaning | Default |
| --- | --- | --- |
| `APCA_API_KEY_ID` | Alpaca API key ID | none |
| `APCA_API_SECRET_KEY` | Alpaca API secret key | none |
| `SIGNALIQ_ALPACA_SECRET_FROM_PROXY` | `true` when the runtime's proxy adds the secret key, so only the key ID is needed | `false` |
| `SIGNALIQ_DATABASE_URL` | SQLite location | `sqlite:///data/signaliq.db` |
| `SIGNALIQ_ALPACA_FEED` | Market data feed | `iex` |
| `SIGNALIQ_ALPACA_ADJUSTMENT` | Corporate-action adjustment | `split` |
| `SIGNALIQ_HISTORY_START` | First date to request | `2016-01-01` |
| `SIGNALIQ_SYMBOLS` | Comma-separated symbol list | the ten symbols |
| `SIGNALIQ_HTTP_TIMEOUT_SECONDS` | Request timeout | `30` |
| `SIGNALIQ_HTTP_MAX_RETRIES` | Retries after a failed request | `5` |
| `SIGNALIQ_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` | `INFO` |

## Tests

```bash
pytest                          # everything
pytest -m "not integration"     # offline tests only
pytest -m integration -v        # live Alpaca tests only
```

The offline tests replace Alpaca with a mocked transport and need no network or
credentials. Their bars are synthetic fixtures with invented values under fake tickers
(`TESTA`, `TESTB`). They are test data only and say nothing about real markets.

The live tests in `tests/test_integration_alpaca.py` run only when credentials are
configured. They make a few small real requests (AAPL daily bars of January 2024, the
same month for all ten symbols in one request, and Thanksgiving week 2024) and check
authentication, rejection of wrong credentials, the response structure, pagination,
every configured symbol, a holiday and an early close, idempotent storage and the API.
Without credentials they are reported as skipped.

## Verification status

What was actually executed for this milestone, on Linux with Python 3.12.3, against the
real Alpaca Market Data API. The live steps were run on 2026-11-17 (UTC) in a managed
runtime where the proxy supplies the secret key.

| # | Completion criterion | Status |
| --- | --- | --- |
| 1 | The application starts locally | Executed: `uvicorn` started on the real database, on a database file that did not exist yet, and again after a restart |
| 2 | Alpaca authentication works with valid credentials | Executed: live tests and real ingestion authenticated through `AlpacaClient`; a wrong key ID was rejected with a readable error; with no credentials the commands stop with exit code 2 |
| 3 | Historical data is retrieved from the specified feed | Executed: all ten symbols downloaded with `feed=iex`, `adjustment=split`, `1Day`: 15,861 bars (see [Data coverage](#data-coverage)) |
| 4 | Pagination is implemented and tested | Executed against a mocked provider and live: SPY's full history in 7 pages of 250 and NVDA's in 4 pages of 400 were identical, bar for bar, to the stored rows; all ten symbols in one request arrived over 6 pages |
| 5 | Data is stored in SQLite | Executed: 15,861 bars, 22 ingestion runs and their check rows in `data/signaliq.db`; SQLite's integrity and foreign-key checks report no problem |
| 6 | Repeated imports do not create duplicates | Executed: a second `ingest.py --all` inserted 0 bars and found all 15,861 unchanged, with 0 conflicts |
| 7 | Validation catches errors and never overwrites | Executed in tests, and on a copy of the real database: a changed close was reported as a `conflicting_record` with both values and the stored value was kept; a deleted bar was downloaded again |
| 8 | API endpoints return stored records | Executed: 52 requests against the running server covering every endpoint, the symbol, timeframe and date filters, paging and invalid input; identical results after a restart |
| 9 | All automated tests pass | 491 passed, including the 6 live tests. With no credentials configured: 485 passed, 6 skipped |
| 10 | Credentials are not exposed | Executed: the key ID does not occur in any API response, in the server log, or in the ingestion output at `DEBUG` level; in this runtime the application never holds the secret key |
| 11 | README instructions reproduce the setup | Executed on Linux: install, database, ingest, validate, inspect, API and tests. The Windows commands were not executed |

Independent checks, made with tools outside the application (plain SQL and the
`exchange_calendars` library, which is not a dependency of this project):

- every stored bar: positive prices, low <= open, close <= high, VWAP inside the range,
  non-negative whole-number volume, a UTC timestamp at midnight New York time, no
  duplicate key;
- the session dates of each symbol against the independent calendar;
- raw against split-adjusted closes around the seven splits inside the stored range.

**Not verified.** HTTP 429 and 5xx handling were exercised with a mocked provider only;
the real API never returned them during these runs (its rate-limit header showed 199
requests still available after a request). A real network failure was simulated with an
unreachable address: the run was recorded as failed and the rest of the batch was not
attempted. The Windows commands were not run. Running with both keys in a `.env` file
(without a proxy) is covered by the offline tests but was not run against the real API
here, because the secret key is not available to this runtime by design.

**Open finding.** `python scripts/validate.py` reports 1 failure and 10 warnings on the
real data and exits with code 1. All eleven are provider coverage, described under
[What the provider does not have](#what-the-provider-does-not-have). No stored bar fails
a row-level check.

## Known limitations

- **History starts on 2020-07-27, not 2016.** See [Data coverage](#data-coverage).
  The older history exists on Alpaca's consolidated SIP feed (see
  [The SIP feed](#the-sip-feed-not-used)), which this milestone does not use. SignalIQ
  never switches feeds on its own.
- **Validation is permanently "failed" for SPY** because of the isolated 2018-11-01 bar.
  Removing a stored bar, or moving `SIGNALIQ_HISTORY_START` to 2020-07-27 so that the
  check window starts there, is a decision for the owner of the data and has not been
  taken. There is no command to delete or acknowledge a finding.
- **IEX data is one exchange.** See [Data coverage](#data-coverage). Do not read IEX
  volume as market volume.
- **The database is local.** It is not in the repository and is lost with the machine or
  cloud session that holds it. Recreating it takes one command and a few seconds.
- **Daily bars only.** The database and API accept any timeframe, but ingestion and gap
  detection are implemented for `1Day`.
- **Symbol changes.** Renamed tickers depend on Alpaca's default symbol mapping. For
  META (FB before June 2022) the history arrives unbroken under META; other renames
  have not been looked at.
- **Split adjustments change history.** After a new split, Alpaca returns re-adjusted
  prices for all earlier bars. A later import will record these as conflicts and keep
  the stored values. There is no command yet to review and accept such revisions.
- **Unscheduled market closures.** A future one-off closure is unknown to the calendar and
  would be reported as a missing session until it is added to `UNSCHEDULED_CLOSURES` in
  `app/data/validation.py`.
- **Full re-download each run.** Every import requests the whole window again. This is
  cheap for daily bars and lets conflicts be detected, but it will not scale to intraday.
- **No schema migrations.** Changing a table later needs a migration tool or a new
  database file.
- **Test client notice.** The newest Starlette test client recommends the `httpx2`
  package and prints a deprecation notice when it runs on HTTPX. The project uses HTTPX
  as specified, the notice is silenced in `pyproject.toml`, and Starlette is pinned so
  the tests keep working.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Alpaca API credentials are not configured` | Create `.env` from `.env.example` and set both keys. Placeholder values count as missing |
| `Alpaca rejected the API credentials (HTTP 401)` | Check both keys for typos or extra spaces, or generate new keys. Where a proxy supplies the secret key, check the key ID and that the secret is saved for `data.alpaca.markets` |
| Live tests are skipped | No credentials are configured. Set both keys, or the key ID and `SIGNALIQ_ALPACA_SECRET_FROM_PROXY=true` |
| `validate.py` ends with `RESULT: FAILED` for SPY only | Expected with the IEX feed: see [What the provider does not have](#what-the-provider-does-not-have) |
| `Alpaca refused the request (HTTP 403)` | The keys are not valid, or your Alpaca plan does not allow the requested feed. Keep `SIGNALIQ_ALPACA_FEED=iex` on the free plan |
| `Alpaca rate limit still exceeded` | Wait a minute and run the command again |
| `A proxy refused the connection to Alpaca` | Your network blocks `data.alpaca.markets` |
| `database not found` | Run `python -m app.database` |
| `ModuleNotFoundError: No module named 'app'` | Run the command from the `signaliq` folder with the virtual environment activated |
| `Symbol 'X' is not in the configured universe` | Use one of the ten symbols, or change `SIGNALIQ_SYMBOLS` |

## Project structure

```
signaliq/
├── app/
│   ├── config.py            settings from .env, logging
│   ├── database.py          engine, sessions, initialisation command
│   ├── models.py            the three tables and the UTC timestamp type
│   ├── data/
│   │   ├── alpaca_client.py  HTTP client: pagination, retries, rate limits, errors
│   │   ├── ingestion.py      download, validate, store, record history
│   │   └── validation.py     checks, trading calendar, session coverage, stored-bar validation
│   ├── api/
│   │   └── routes.py         the five endpoints
│   └── main.py              FastAPI application
├── scripts/
│   ├── ingest.py            download bars
│   ├── validate.py          validate stored bars
│   └── inspect_db.py        database statistics
├── tests/                   offline tests plus one optional live test file
├── data/                    the SQLite database is created here
├── .env.example
├── requirements.txt
└── pyproject.toml
```

Optional code style check: `pip install ruff`, then `ruff check .` and `ruff format .`.

## Next steps: Milestone 2

Proposed scope: intraday data, automatic updates and operating the data quality process.
Still no strategies or signals.

1. **Decide how to treat the SPY stray bar** (keep it flagged, move the history start to
   2020-07-27, or add a reviewed way to acknowledge a finding), and whether six years of
   IEX history are enough or the consolidated SIP feed is needed.
2. **Add schema migrations** (Alembic) before any table changes.
3. **Incremental updates.** Download only from the last stored bar, keep a periodic full
   re-check for revisions, and add a scheduled daily update after 20:15 New York time.
4. **Conflict review.** Add a command that lists conflicts, shows stored and incoming
   values side by side, and applies an accepted revision with an audit record. Handle
   the re-adjustment after a split as one reviewed event, not thousands of conflicts.
5. **Intraday ingestion** for 1Min, 5Min, 15Min and 1Hour: store page by page, use the
   calendar's session closing times (early closes are already known to it) and define
   when an intraday bar is complete.
6. **Intraday gap detection** that separates regular and extended hours and allows for
   minutes without IEX trades.
7. **Universe management.** Load symbol metadata from Alpaca's assets endpoint, track
   renames and delistings, and grow the list beyond ten symbols.
8. **Feed decision.** Decide whether to add the consolidated SIP feed, which would extend
   the history back to 2016 (see [The SIP feed](#the-sip-feed-not-used)). If so, make the
   feed part of the bar key so two feeds can be stored side by side, and end each request
   at least 15 minutes in the past, because this account is refused more recent SIP data.
