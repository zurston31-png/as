#!/usr/bin/env bash
#
# Pull, rebuild and restart the bot when the tracked branch moves.
#
# Run from a systemd timer (deploy/systemd/memecoin-bot-update.*). Safe to
# run when nothing has changed: it fetches, sees the same commit, and
# exits without touching the service.
#
# Both documented deployment paths are supported and auto-detected:
#
#   docker   - deploy/vps_setup.md section 1. Rebuilds the image, then
#              swaps the container.
#   systemd  - deploy/vps_setup.md section 2. Reinstalls requirements into
#              the venv, then restarts the unit.
#
# Override with MODE=docker or MODE=systemd if detection guesses wrong.
#
# WHAT THIS REFUSES TO DO, AND WHY
#
# Automatic deployment means code reaches a running trading bot with no
# human between commit and production. For this project that is a bigger
# deal than usual, because the value of the whole exercise is one clean
# dataset collected under ONE frozen configuration. So the update aborts,
# rather than proceeding, when any of these is true:
#
#   * the strategy version hash would change. That is the freeze, enforced
#     mechanically: a new hash splits the dataset, and no convenience is
#     worth silently doing that at 4am. The check runs against the NEW
#     code BEFORE the running service is replaced.
#   * LIVE_TRADING or LIVE_EXECUTION_ACKNOWLEDGED is not false in the new
#     code's settings. Paper-only is not a preference.
#   * the pull is not a fast-forward. Divergence means someone did
#     something by hand and an automated merge would hide it.
#   * the build or dependency install fails. The old service keeps running.
#   * the service is not healthy within HEALTH_TIMEOUT. The previous commit
#     is restored, rebuilt and restarted automatically.
#
# WHY IT TRACKS A DEPLOYED MARKER AND NOT JUST git HEAD
#
# Comparing HEAD to origin is not enough. `git pull` moves the checkout
# without rebuilding, so a host where someone pulled by hand has new code
# on disk and the OLD build still serving. HEAD == origin, so a
# HEAD-only check reports "up to date" forever and the box stays stale
# behind a timer that says otherwise. That happened on the first install.
#
# So the commit that was last successfully DEPLOYED is recorded in
# $DEPLOYED_MARKER after the health check passes, and an update runs when
# either the branch moved or the marker does not match HEAD. A missing
# marker means "unknown", which deploys - correct on first install, where
# the running artifact genuinely cannot be verified.
#
# The database is backed up before any restart. On a correctly configured
# host it also lives outside the clone, so it is not at risk either way.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/memecoin-bot-live}"
BRANCH="${BRANCH:-claude/memecoin-trading-bot-im07pf}"
DATA_DIR="${DATA_DIR:-/data}"
DB_PATH="${DB_PATH:-$DATA_DIR/memecoin_bot.db}"
BACKUP_DIR="${BACKUP_DIR:-$DATA_DIR/deploy-backups}"
CONTAINER="${CONTAINER:-memecoin-bot}"
SERVICE="${SERVICE:-memecoin-bot}"
RUN_USER="${RUN_USER:-botuser}"
VENV="${VENV:-venv}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"
BACKUPS_KEPT="${BACKUPS_KEPT:-10}"
# Outside the clone on purpose: a marker inside it would be destroyed by
# the same reset that rolls a bad deploy back.
DEPLOYED_MARKER="${DEPLOYED_MARKER:-$DATA_DIR/.deployed_commit}"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { log "ABORT: $*"; exit 1; }

cd "$REPO_DIR" || die "no such repo directory: $REPO_DIR"

# --- which deployment is this? -----------------------------------------
if [ -z "${MODE:-}" ]; then
    if [ -f docker-compose.yml ] && command -v docker >/dev/null 2>&1 \
       && docker compose ps --quiet 2>/dev/null | grep -q .; then
        MODE=docker
    elif [ -x "$VENV/bin/python" ]; then
        MODE=systemd
    else
        die "cannot tell whether this is a docker or systemd deployment - set MODE"
    fi
fi
log "deployment mode: $MODE"

# --- is there anything to do? ------------------------------------------
git fetch --quiet origin "$BRANCH" || die "git fetch failed"
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"

DEPLOYED=""
[ -f "$DEPLOYED_MARKER" ] && DEPLOYED="$(tr -d '[:space:]' < "$DEPLOYED_MARKER")"

if [ "$LOCAL" = "$REMOTE" ] && [ "$DEPLOYED" = "$LOCAL" ]; then
    log "up to date at ${LOCAL:0:7} and deployed, nothing to do"
    exit 0
fi

if ! git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
    die "origin/$BRANCH is not a fast-forward from ${LOCAL:0:7} - resolve by hand"
fi

if [ "$LOCAL" != "$REMOTE" ]; then
    log "branch moved: ${LOCAL:0:7} -> ${REMOTE:0:7}"
elif [ -z "$DEPLOYED" ]; then
    log "no deployed marker at $DEPLOYED_MARKER - cannot verify what is running,"
    log "so redeploying ${LOCAL:0:7} to make the marker true"
else
    log "checkout is ${LOCAL:0:7} but ${DEPLOYED:0:7} was last deployed"
    log "(someone pulled without rebuilding) - redeploying"
fi

PREVIOUS="$LOCAL"

if ! git diff --quiet || ! git diff --cached --quiet; then
    # docker-compose.yml may be intentionally skip-worktree'd on a host, so
    # a dirty tree here means something ELSE was edited in place.
    log "WARNING: working tree has uncommitted changes; they will be preserved by"
    log "         the fast-forward but may conflict. Files:"
    git diff --name-only | sed 's/^/           /'
fi

# --- reading the strategy version and the live flags --------------------
# In docker mode these run in a THROWAWAY container off the freshly built
# image; in systemd mode they run from the updated checkout. Either way the
# answer describes what is ABOUT to be deployed, not what is running. The
# image has a CMD and no ENTRYPOINT, so passing a command simply replaces
# the uvicorn CMD.
PROBE_VERSION='from app.strategy.version import current_label; print(current_label())'
PROBE_FLAGS='from app.config import settings; print(settings.LIVE_TRADING, settings.LIVE_EXECUTION_ACKNOWLEDGED)'

probe_new() {
    if [ "$MODE" = docker ]; then
        docker compose run --rm --no-deps bot python -c "$1" 2>/dev/null
    else
        sudo -u "$RUN_USER" "$VENV/bin/python" -c "$1" 2>/dev/null
    fi | tr -d '\r' | tail -n 1
}

if [ "$MODE" = docker ]; then
    RUNNING_VERSION="$(docker exec "$CONTAINER" python -c "$PROBE_VERSION" 2>/dev/null | tr -d '\r' | tail -n 1)"
else
    RUNNING_VERSION="$(sudo -u "$RUN_USER" "$VENV/bin/python" -c "$PROBE_VERSION" 2>/dev/null | tr -d '\r' | tail -n 1)"
fi

if [ -z "$RUNNING_VERSION" ]; then
    log "WARNING: could not read the running strategy version (service down?)."
    log "         The freeze check needs a baseline, so this update is skipped."
    log "         Start the bot, or deploy by hand, then the timer resumes."
    exit 1
fi
log "running strategy version: $RUNNING_VERSION"

# --- back up before anything is changed --------------------------------
mkdir -p "$BACKUP_DIR"
if [ -f "$DB_PATH" ]; then
    BACKUP="$BACKUP_DIR/memecoin_bot.db.$(date -u '+%Y%m%dT%H%M%SZ').${PREVIOUS:0:7}"
    cp "$DB_PATH" "$BACKUP" || die "database backup failed - refusing to update"
    log "backed up database to $BACKUP"
    ls -1t "$BACKUP_DIR"/memecoin_bot.db.* 2>/dev/null \
        | tail -n +$((BACKUPS_KEPT + 1)) | xargs -r rm -f
else
    log "WARNING: no database at $DB_PATH - continuing, but check DATABASE_URL"
fi

# --- pull and prepare (old service still serving) -----------------------
git merge --ff-only "origin/$BRANCH" --quiet || die "fast-forward merge failed"
log "now at $(git rev-parse --short HEAD)"

rollback_checkout() {
    git reset --hard "$PREVIOUS" --quiet
}

prepare() {
    if [ "$MODE" = docker ]; then
        docker compose build
    else
        sudo -u "$RUN_USER" "$VENV/bin/pip" install --quiet -r requirements.txt
    fi
}

restart() {
    if [ "$MODE" = docker ]; then
        docker compose up -d
    else
        systemctl restart "$SERVICE"
    fi
}

if ! prepare; then
    log "build/install failed - rolling the checkout back, old service still running"
    rollback_checkout
    die "prepare failed at ${REMOTE:0:7}"
fi

# --- gate on the NEW code before it replaces the running service --------
NEW_VERSION="$(probe_new "$PROBE_VERSION")"
if [ -z "$NEW_VERSION" ]; then
    rollback_checkout
    prepare >/dev/null 2>&1
    die "could not read the strategy version from the new build"
fi

if [ "$NEW_VERSION" != "$RUNNING_VERSION" ]; then
    log "strategy version would change: $RUNNING_VERSION -> $NEW_VERSION"
    log "That splits the collection dataset. Not deploying automatically."
    log "If the change is intended, deploy it by hand and start a new run."
    rollback_checkout
    prepare >/dev/null 2>&1
    die "frozen strategy version would change"
fi
log "strategy version unchanged ($NEW_VERSION)"

LIVE_FLAGS="$(probe_new "$PROBE_FLAGS")"
if [ "$LIVE_FLAGS" != "False False" ]; then
    rollback_checkout
    prepare >/dev/null 2>&1
    die "new build reports LIVE_TRADING/LIVE_EXECUTION_ACKNOWLEDGED = '$LIVE_FLAGS', expected 'False False'"
fi
log "paper-only confirmed"

# --- swap ---------------------------------------------------------------
log "restarting"
restart

deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null | grep -q '"status":"ok"'; then
        healthy=1
        break
    fi
    sleep 3
done

if [ "$healthy" -ne 1 ]; then
    log "NOT healthy within ${HEALTH_TIMEOUT}s - rolling back to ${PREVIOUS:0:7}"
    rollback_checkout
    if prepare && restart; then
        printf '%s\n' "$PREVIOUS" > "$DEPLOYED_MARKER"
    fi
    die "rolled back to ${PREVIOUS:0:7}; investigate ${REMOTE:0:7}"
fi

printf '%s\n' "$(git rev-parse HEAD)" > "$DEPLOYED_MARKER"
log "healthy at $(git rev-parse --short HEAD) - update complete"
