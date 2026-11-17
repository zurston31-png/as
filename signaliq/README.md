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
- [How the data is handled](#how-the-data-is-handled)
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
bars that are already stored are skipped.

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
| Inspect the database | `python scripts/inspect_db.py` (add `--json` for JSON output) |
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
| `GET /symbols` | The configured symbols with the stored bar count and date range of each |
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

`GET /ingestion/runs` accepts `symbol`, `status`, `limit` and `offset`.
`GET /quality/report` accepts `symbol`, `stage` (`ingestion` or `stored`) and `limit`.

Invalid input returns a clear message: an unknown symbol returns 404 with the list of
supported symbols, and a malformed date, timeframe or paging value returns 422.

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
sessions.

- A missing session between stored bars is a **failure**.
- Sessions missing before the first stored bar are a **warning**, because the provider's
  history for a feed may start later than 2016-01-01.
- Sessions missing after the last stored bar are a **warning** that the data needs
  updating.

## Database

SQLite file: `data/signaliq.db`.

| Table | Contents |
| --- | --- |
| `bars` | One row per bar: symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count, source, feed, adjustment. Primary key: symbol, timeframe, timestamp. Each row also records the ingestion run that stored it and when. |
| `ingestion_runs` | One row per symbol and download attempt: requested window, feed, status, counters for fetched, inserted, unchanged, conflicting, rejected and duplicate bars, missing sessions, and any error message. |
| `data_quality_checks` | Check outcomes from ingestion and from `validate.py`. A `passed` row summarises one check; a `failed` or `warning` row describes one finding with the offending values. |

## Configuration

Settings are read from `.env` and from environment variables. Only the two credentials
are required.

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
configured. They make a few small real requests for AAPL daily bars of January 2024 and
check authentication, the response structure, pagination, idempotent storage and the
API. Without credentials they are reported as skipped.

## Verification status

What was actually executed for this milestone, and what was not.

| # | Completion criterion | Status |
| --- | --- | --- |
| 1 | The application starts locally | Executed: `uvicorn` started and every endpoint answered over HTTP |
| 2 | Alpaca authentication works with valid credentials | Executed: the live tests authenticated against `data.alpaca.markets` through `AlpacaClient` (key ID from the environment, secret key added by the runtime's proxy), and deliberately wrong credentials were rejected |
| 3 | Historical data is retrieved from the specified feed | Executed: 21 AAPL daily bars for January 2024 were retrieved with `feed=iex`, `adjustment=split` and passed every validation check. The full history from 2016 and the other nine symbols were not downloaded |
| 4 | Pagination is implemented and tested | Executed against a mocked provider and live (page size 5 over the same month returned the same bars as one request) |
| 5 | Data is stored in SQLite | Executed in tests and with the real commands on a scratch database |
| 6 | Repeated imports do not create duplicates | Executed in tests, including five repeated imports |
| 7 | Validation catches deliberately introduced errors | Executed in tests and by corrupting rows in a scratch database and running `validate.py` |
| 8 | API endpoints return stored records | Executed in tests and against the running server |
| 9 | All automated tests pass | 442 passed on Python 3.12.3, including the 4 live tests; 438 passed with the live tests skipped when no credentials are configured |
| 10 | README instructions reproduce the setup | Executed on Linux in a fresh virtual environment: install, database, validate, inspect, API and tests. `ingest.py` has not been run against the real API. The Windows commands were not executed |

To repeat criteria 2, 3 and the live half of 4 on your machine, add your keys to `.env`
and run:

```bash
pytest -m integration -v
python scripts/ingest.py --symbol AAPL
python scripts/ingest.py --symbol AAPL     # second run: 0 inserted, all unchanged
python scripts/validate.py --symbol AAPL
```

## Known limitations

- **Only a small live run so far.** The live tests fetched and stored one month of AAPL
  daily bars. The full ingestion from 2016 for all ten symbols has not been run against
  the real API, so it may still show differences that need a fix.
- **IEX data is one exchange.** See [How the data is handled](#how-the-data-is-handled).
  Do not read IEX volume as market volume.
- **History before the feed's first bar.** How far back Alpaca's IEX history goes has not
  been measured. If it starts after 2016-01-01 the difference is reported as a warning.
- **Daily bars only.** The database and API accept any timeframe, but ingestion and gap
  detection are implemented for `1Day`.
- **Symbol changes.** Renamed tickers (META traded as FB before June 2022) depend on
  Alpaca's default symbol mapping. This has not been verified.
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
| `Alpaca rejected the API credentials (HTTP 401)` | Check both keys for typos or extra spaces, or generate new keys |
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
│   │   └── validation.py     checks, trading calendar, stored-bar validation
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

1. **Close the live verification.** Run the commands under
   [Verification status](#verification-status), then download all ten symbols and read
   `GET /quality/report`. Record where the IEX history really starts for each symbol and
   confirm the META history before June 2022.
2. **Add schema migrations** (Alembic) before any table changes.
3. **Incremental updates.** Download only from the last stored bar, keep a periodic full
   re-check for revisions, and add a scheduled daily update after 20:15 New York time.
4. **Conflict review.** Add a command that lists conflicts, shows stored and incoming
   values side by side, and applies an accepted revision with an audit record. Handle
   the re-adjustment after a split as one reviewed event, not thousands of conflicts.
5. **Intraday ingestion** for 1Min, 5Min, 15Min and 1Hour: store page by page, extend the
   calendar with session hours and early closes, and define when an intraday bar is
   complete.
6. **Intraday gap detection** that separates regular and extended hours and allows for
   minutes without IEX trades.
7. **Universe management.** Load symbol metadata from Alpaca's assets endpoint, track
   renames and delistings, and grow the list beyond ten symbols.
8. **Feed decision.** Decide whether to add the consolidated SIP feed. If so, make the
   feed part of the bar key so two feeds can be stored side by side.
