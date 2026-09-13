# VPS Deployment Guide

Two supported paths: **Docker** (recommended) or **systemd + venv**. Both
assume a fresh Ubuntu 22.04+ VPS with a non-root sudo user.

## 0. Local dev: exposing the webhook with ngrok

Before renting a VPS, you can develop entirely on your laptop:

```bash
cp .env.example .env        # fill in WEBHOOK_SECRET at minimum
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
uvicorn app.main:app --reload --port 8000

# in a second terminal
ngrok http 8000
```

Use the `https://<random>.ngrok-free.app/webhook/tradingview` URL ngrok
prints as the TradingView alert webhook URL. Free ngrok URLs rotate on
restart — fine for development, not for a 24/7 bot, hence the VPS step below.

## 1. Docker deployment (recommended)

```bash
# on the VPS
# docker-compose-v2, NOT docker-compose-plugin: the latter is the package
# name in Docker's own apt repo, and is not in Ubuntu's. Getting it wrong
# is worse than it sounds - apt installs nothing at all when one name in
# the list is unresolvable, so docker.io silently does not arrive either
# and the next command fails with "docker: command not found".
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker

git clone <your-fork-url> memecoin-bot
cd memecoin-bot
cp .env.example .env
nano .env   # set WEBHOOK_SECRET, risk limits, notification tokens, etc.

# Point the database and backups at a host path OUTSIDE this clone, so a
# redeploy - unpacking a new version into a fresh directory - does not
# start a new history. The compose file below expects exactly this path.
mkdir -p /data
# in .env, set:
#   DATABASE_URL=sqlite:////data/memecoin_bot.db   (four slashes)
#   BACKUP_DIR=/data/backups
#   BACKUP_DIR_IS_PERSISTENT=true
# docker-compose.yml's volumes entry (`/data:/data`) must match whatever
# absolute path you choose here - it is not ./data relative to this clone,
# on purpose. Getting the two out of sync is silent: the container starts
# fine and just writes to a location that disappears with it, so a trade
# recorded one run is gone on the next `docker compose up`.

docker compose up -d --build
docker compose logs -f bot   # confirm it started in PAPER mode
curl http://localhost:8000/health
```

`docker-compose.yml` sets `restart: unless-stopped`, so the container comes
back automatically after a crash or VPS reboot — no manual intervention
needed, matching the "runs 24/7 unattended" requirement.

### Production TLS / domain

TradingView requires HTTPS for webhooks. Point a domain's A record at the
VPS, then either:

- **Caddy** (simplest, auto-HTTPS via Let's Encrypt): uncomment the `caddy`
  service in `docker-compose.yml`, add a `Caddyfile`:
  ```
  your-domain.example {
      reverse_proxy bot:8000
  }
  ```
- **nginx + certbot**: standard reverse-proxy-to-127.0.0.1:8000 setup with
  `certbot --nginx`.

Either way, the final TradingView webhook URL becomes:
`https://your-domain.example/webhook/tradingview`

## 2. systemd + venv deployment (no Docker)

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
sudo useradd -r -m -d /opt/memecoin-bot botuser

sudo -u botuser git clone <your-fork-url> /opt/memecoin-bot
cd /opt/memecoin-bot
sudo -u botuser python3.11 -m venv venv
sudo -u botuser ./venv/bin/pip install -r requirements.txt
sudo -u botuser cp .env.example .env
sudo -u botuser nano .env
sudo -u botuser ./venv/bin/python scripts/init_db.py

sudo mkdir -p /var/log/memecoin-bot && sudo chown botuser:botuser /var/log/memecoin-bot

sudo cp deploy/systemd/memecoin-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now memecoin-bot
sudo systemctl status memecoin-bot
journalctl -u memecoin-bot -f
```

`Restart=always` in the unit file gives you the same auto-restart-on-crash
guarantee as Docker's restart policy.

## 3. Firewall

Only expose what's needed — the reverse proxy (80/443) if using one, or
8000 directly if TradingView will hit the VPS IP over HTTPS via some other
TLS terminator you control. Do not expose the dashboard/webhook port without
TLS in production; TradingView will not deliver webhooks to plain HTTP.

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp
sudo ufw enable
```

## 4. Updating a deployment that is already running

The guide above installs. This section upgrades — a different and more
dangerous operation, because the thing you must not lose is the database.

> **Set up the timer once and skip most of this section.**
> [Automatic updates](#automatic-updates-systemd-timer) below installs a
> systemd timer that does everything here every 15 minutes, on either
> deployment path, and rolls itself back if the new build is unhealthy.
> The manual steps stay documented because you still need them the first
> time, when the automatic update deliberately refuses (a strategy-version
> change), and when something has gone wrong enough that you want to drive
> it yourself.

**Every accounting fix since the first deployment is on the READ side.**
`app/execution/paper.py` and `app/execution/fill_model.py` — the code that
computes `fee_usd` and `execution_cost_pct` and writes them to the trade
row — are unchanged. The defects were in `app/analysis/*`, which reads
those raw columns and derives slippage, cost rates and post-mortems from
them. So updating the code **re-derives the corrected figures from the
trades you already have**. You do not lose your sample and you do not
need to reset the run.

New columns (`price_ticks_observed`, `fill_estimated_from_quote`,
`idempotency_key`) are added automatically on the next start by
`apply_additive_migrations()`, called from `init_db()` in the app's
startup. Existing rows get `NULL`, which reads as "not recorded at the
time" rather than a fabricated zero. There is no manual migration step.

### Back up first — always

```bash
# adjust the path to wherever DATABASE_URL points on your host
sudo cp /data/memecoin_bot.db /data/memecoin_bot.db.pre-upgrade-$(date +%F)
```

Take this copy even though the upgrade is additive. It costs a second and
it is the only thing standing between a bad `rm -rf` and the whole
collection run.

### Docker path

```bash
cd /opt/memecoin-bot          # wherever the clone lives
git fetch origin
git checkout claude/memecoin-trading-bot-im07pf
git pull --ff-only

docker compose down
docker compose build
docker compose up -d
docker compose logs -f --tail=50
```

`.env` and the database live outside the clone (section 1), so neither is
touched by `git pull`. If `git pull` reports local changes you did not
make, stop and look — do not force past it.

### systemd + venv path

```bash
cd /opt/memecoin-bot
sudo -u botuser git fetch origin
sudo -u botuser git checkout claude/memecoin-trading-bot-im07pf
sudo -u botuser git pull --ff-only
sudo -u botuser ./venv/bin/pip install -r requirements.txt

sudo systemctl restart memecoin-bot
sudo systemctl status memecoin-bot --no-pager
sudo journalctl -u memecoin-bot -f -n 50
```

### Verify the upgrade landed

```bash
git rev-parse --short HEAD            # should match the version you deployed
./venv/bin/python scripts/research.py preflight
./venv/bin/python scripts/performance_report.py
```

Two things to look for in the report afterwards:

- the per-attribute breakdowns (signal score, market quality, entry
  liquidity, holding time) should now show buckets instead of
  "N trade(s) with this value not recorded" — those attributes were
  always being recorded, they were previously read off the wrong trade
  leg
- execution cost and slippage should be internally consistent: cost is
  never smaller than the fee component, and the reported rate times the
  costed notional reproduces the reported dollar cost

If the breakdowns still say "not recorded" after the restart, the new
code is not actually running — check `git rev-parse HEAD` and that the
service restarted, rather than assuming the data is missing.

### Automatic updates (systemd timer)

`deploy/auto_update.sh` plus the two units in `deploy/systemd/` poll the
branch every 15 minutes and redeploy when it moves. It handles **both**
deployment paths in this guide and detects which one it is on:

| Mode | Detected by | Prepare step | Restart |
|---|---|---|---|
| `docker` (section 1) | a `docker-compose.yml` and a running compose project | `docker compose build` | `docker compose up -d` |
| `systemd` (section 2) | an executable `$VENV/bin/python` | `pip install -r requirements.txt` into the venv | `systemctl restart` |

Force it with `MODE=docker` or `MODE=systemd` if detection guesses wrong.

Install on the VPS — identical for both paths:

```bash
cd /root/memecoin-bot-live          # or /opt/memecoin-bot
chmod +x deploy/auto_update.sh

# dry run FIRST - it exits harmlessly when there is nothing to pull
./deploy/auto_update.sh

sudo cp deploy/systemd/memecoin-bot-update.service /etc/systemd/system/
sudo cp deploy/systemd/memecoin-bot-update.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now memecoin-bot-update.timer

systemctl list-timers memecoin-bot-update.timer
tail -f /var/log/memecoin-bot-update.log
```

The units assume the clone is at `/root/memecoin-bot-live`. On a
systemd + venv host it is usually `/opt/memecoin-bot` — edit
`WorkingDirectory` and `ExecStart` to match. Everything else is an
environment variable with a sensible default: `REPO_DIR`, `BRANCH`,
`MODE`, `DATA_DIR`, `DB_PATH`, `BACKUP_DIR`, `CONTAINER`, `SERVICE`,
`RUN_USER`, `VENV`, `HEALTH_URL`, `HEALTH_TIMEOUT`, `BACKUPS_KEPT`.

A systemd + venv host also needs the updater to run as root (to restart
the unit) while installing packages as the service account — that is what
`RUN_USER` is for, and it defaults to `botuser` to match section 2.

**What it refuses to deploy.** Automatic deployment puts code on a running
bot with nobody in between, so the script is built to abort rather than
proceed:

| Condition | What happens |
|---|---|
| Strategy version hash would change | Aborts, checkout reset. A new hash splits the collection dataset — that decision is never made by a timer. |
| `LIVE_TRADING` / `LIVE_EXECUTION_ACKNOWLEDGED` not both false | Aborts, checkout reset. |
| Not a fast-forward | Aborts. Divergence means someone worked by hand; an automatic merge would bury it. |
| Build or dependency install fails | Checkout reset, old service never stopped. |
| Service unhealthy within 90s | Previous commit restored, rebuilt/reinstalled, and brought back up automatically. |

The version and paper-only checks run against the **new code before it
replaces the running service** — in a throwaway container on the docker
path, from the updated checkout on the systemd path — so a bad deploy is
caught before it serves anything.

The database is copied to `/data/deploy-backups/` before any restart (last
10 kept) and lives outside the clone, so the update cannot touch it.

**It tracks what was deployed, not just what was pulled.** Comparing
`HEAD` to `origin` alone is not enough: `git pull` moves the checkout
without rebuilding, so a host where someone pulled by hand has new code on
disk and the old build still serving — and a `HEAD`-only check would
report "up to date" forever while the box stayed stale. The commit last
successfully deployed is recorded in `/data/.deployed_commit` after the
health check passes, and an update runs when the branch moved **or** that
marker disagrees with `HEAD`. On first install the marker is absent, so
the first timer run redeploys once to make it true — that is expected, not
a fault.

**Turning it off:**

```bash
sudo systemctl disable --now memecoin-bot-update.timer
```

### Rolling back

```bash
git checkout <previous-sha>
# docker: docker compose down && docker compose build && docker compose up -d
# systemd: sudo systemctl restart memecoin-bot
```

The database is forward-compatible with older code: the added columns are
nullable and nothing older reads them. Restore the pre-upgrade copy only
if you have a specific reason to — rolling the data back discards any
trades recorded since the upgrade.

## 5. Keep it paper-only

There is no going-live step. `LIVE_TRADING` and
`LIVE_EXECUTION_ACKNOWLEDGED` stay `false`: no wallet keys, no real funds,
no live-order execution. The launcher and the operator scripts refuse to
start against an .env with either flag enabled
(`app/safety/paper_only.py`), and `python scripts/research.py preflight`
reports the same check for a running deployment.

See the README's "Paper-run checklist" for what to do with the collection
run instead.
