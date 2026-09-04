#!/usr/bin/env bash
#
# Pull, rebuild and restart the bot when the tracked branch moves.
#
# Run from a systemd timer (deploy/systemd/memecoin-bot-update.*). Safe to
# run when nothing has changed: it fetches, sees the same commit, and
# exits without touching the container.
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
#     image BEFORE the running container is replaced.
#   * LIVE_TRADING or LIVE_EXECUTION_ACKNOWLEDGED is not false in the new
#     image's settings. Paper-only is not a preference.
#   * the pull is not a fast-forward. Divergence means someone did
#     something by hand and an automated merge would hide it.
#   * the build fails. The old container is left running, untouched.
#   * the new container is not healthy within HEALTH_TIMEOUT. The previous
#     commit is restored and rebuilt automatically.
#
# The database is backed up before any restart, and lives on /data outside
# the clone, so it is not at risk from the update itself.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/memecoin-bot-live}"
BRANCH="${BRANCH:-claude/memecoin-trading-bot-im07pf}"
DATA_DIR="${DATA_DIR:-/data}"
DB_PATH="${DB_PATH:-$DATA_DIR/memecoin_bot.db}"
BACKUP_DIR="${BACKUP_DIR:-$DATA_DIR/deploy-backups}"
CONTAINER="${CONTAINER:-memecoin-bot}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"
BACKUPS_KEPT="${BACKUPS_KEPT:-10}"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { log "ABORT: $*"; exit 1; }

cd "$REPO_DIR" || die "no such repo directory: $REPO_DIR"

# --- is there anything to do? ------------------------------------------
git fetch --quiet origin "$BRANCH" || die "git fetch failed"
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"

if [ "$LOCAL" = "$REMOTE" ]; then
    log "up to date at ${LOCAL:0:7}, nothing to do"
    exit 0
fi

if ! git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
    die "origin/$BRANCH is not a fast-forward from ${LOCAL:0:7} - resolve by hand"
fi

log "update available: ${LOCAL:0:7} -> ${REMOTE:0:7}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    # docker-compose.yml is intentionally skip-worktree'd on this host, so
    # a dirty tree here means something ELSE was edited in place.
    log "WARNING: working tree has uncommitted changes; they will be preserved by"
    log "         the fast-forward but may conflict. Files:"
    git diff --name-only | sed 's/^/           /'
fi

# --- record what we can roll back to -----------------------------------
PREVIOUS="$LOCAL"

strategy_version_of_image() {
    # Runs in a THROWAWAY container off the freshly built image, so the
    # answer describes what is about to be deployed, not what is running.
    # The image has a CMD and no ENTRYPOINT, so a command passed here
    # simply replaces the uvicorn CMD - no --entrypoint override needed,
    # and adding one would clear something that does not exist.
    docker compose run --rm --no-deps bot \
        python -c 'from app.strategy.version import current_label; print(current_label())' \
        2>/dev/null | tr -d '\r' | tail -n 1
}

live_flags_of_image() {
    docker compose run --rm --no-deps bot \
        python -c 'from app.config import settings; print(settings.LIVE_TRADING, settings.LIVE_EXECUTION_ACKNOWLEDGED)' \
        2>/dev/null | tr -d '\r' | tail -n 1
}

RUNNING_VERSION="$(docker exec "$CONTAINER" \
    python -c 'from app.strategy.version import current_label; print(current_label())' \
    2>/dev/null | tr -d '\r' | tail -n 1)"
if [ -z "$RUNNING_VERSION" ]; then
    log "WARNING: could not read the running strategy version (container down?)."
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

# --- pull and build (old container still serving) -----------------------
git merge --ff-only "origin/$BRANCH" --quiet || die "fast-forward merge failed"
log "now at $(git rev-parse --short HEAD)"

if ! docker compose build; then
    log "build failed - rolling the checkout back, old container still running"
    git reset --hard "$PREVIOUS" --quiet
    die "build failed at ${REMOTE:0:7}"
fi

# --- gate on the new image BEFORE it replaces the running one -----------
NEW_VERSION="$(strategy_version_of_image)"
if [ -z "$NEW_VERSION" ]; then
    git reset --hard "$PREVIOUS" --quiet
    die "could not read the strategy version from the new image"
fi

if [ "$NEW_VERSION" != "$RUNNING_VERSION" ]; then
    git reset --hard "$PREVIOUS" --quiet
    log "strategy version would change: $RUNNING_VERSION -> $NEW_VERSION"
    log "That splits the collection dataset. Not deploying automatically."
    log "If the change is intended, deploy it by hand and start a new run."
    die "frozen strategy version would change"
fi
log "strategy version unchanged ($NEW_VERSION)"

LIVE_FLAGS="$(live_flags_of_image)"
if [ "$LIVE_FLAGS" != "False False" ]; then
    git reset --hard "$PREVIOUS" --quiet
    die "new image reports LIVE_TRADING/LIVE_EXECUTION_ACKNOWLEDGED = '$LIVE_FLAGS', expected 'False False'"
fi
log "paper-only confirmed"

# --- swap ---------------------------------------------------------------
log "restarting"
docker compose up -d

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
    git reset --hard "$PREVIOUS" --quiet
    docker compose build && docker compose up -d
    die "rolled back to ${PREVIOUS:0:7}; investigate $(git rev-parse --short "$REMOTE")"
fi

log "healthy at $(git rev-parse --short HEAD) - update complete"
