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

## 4. Keep it paper-only

There is no going-live step. `LIVE_TRADING` and
`LIVE_EXECUTION_ACKNOWLEDGED` stay `false`: no wallet keys, no real funds,
no live-order execution. The launcher and the operator scripts refuse to
start against an .env with either flag enabled
(`app/safety/paper_only.py`), and `python scripts/research.py preflight`
reports the same check for a running deployment.

See the README's "Paper-run checklist" for what to do with the collection
run instead.
