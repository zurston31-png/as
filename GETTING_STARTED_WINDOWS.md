# Getting Started on Windows

A no-experience-required guide to running the bot on your own PC.

**Nothing in this guide involves real money.** The bot runs in *paper
trading* mode: it makes pretend trades with pretend money so you can watch
how it behaves. No wallet, no crypto, no card, no accounts.

---

## Step 1: Install Python (one time, ~2 minutes)

Python is the free programming language the bot is written in.

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **Download Python** button
3. Open the file it downloads
4. ⚠️ **On the first screen, tick the box at the bottom that says
   "Add python.exe to PATH".** This is easy to miss and nothing works
   without it.
5. Click **Install Now** and wait for it to finish

> Already have Python? Skip this step. `START_HERE.bat` will tell you if
> something's missing.

---

## Step 2: Download the bot

1. Go to **https://github.com/zurston31-png/as**
2. Click the branch dropdown (it says `main`) and pick
   **`claude/memecoin-trading-bot-im07pf`**
3. Click the green **Code** button → **Download ZIP**
4. Find the ZIP in your Downloads folder, **right-click → Extract All**
5. Pick somewhere easy to find, like your Desktop

You should now have a folder containing `START_HERE.bat`.

---

## Step 3: Start the bot

**Double-click `START_HERE.bat`.**

A black window opens and shows progress. The first run takes a couple of
minutes while it downloads what it needs. It will:

- set itself up
- generate your passwords automatically
- start the bot
- open the dashboard in your browser

When it's done you'll see a **READY** message with your dashboard password.
Write that password down — it's also saved in a file called `.env` inside
the folder.

> **Windows may show a blue "Windows protected your PC" warning.** That
> appears for any downloaded script that isn't from a big company. Click
> **More info → Run anyway** if you're comfortable; the script is plain
> text and you can open it in Notepad to read exactly what it does.

**To stop the bot:** close the black window, or press `Ctrl+C` in it.

---

## Step 4: Look around the dashboard

Your browser should open to **http://127.0.0.1:8000** and ask for a
username and password:

- Username: `admin`
- Password: the one shown in the black window

You'll see an empty dashboard showing **PAPER** mode and a pretend
**$1,000.00** balance. Empty is correct — the bot hasn't been told about
any trades yet.

---

## Step 5: Watch it make a decision

**Double-click `SEND_TEST_SIGNAL.bat`** (leave the bot running).

This pretends to be TradingView sending the bot a "buy" alert.

Pick option **1**, then refresh the dashboard.

You'll see the signal appear, and under **Risk / Rejection Events**:

> `rug_check_rejected — no on-chain token address supplied with signal`

**That's the bot working correctly, not an error.** It refused to buy
because it couldn't run scam checks on the token. The bot is built to
refuse anything it can't verify.

Try option **2** with a real token address copied from
**https://dexscreener.com** to watch the real scam checks run. Most
memecoins fail them — thin liquidity, too few holders owning too much,
mint authority still active. That's the filter doing its job.

---

## What you have now

A working bot on your own computer that:

- receives trading alerts
- screens tokens for scams and rug-pulls before buying
- enforces strict risk limits (max 2% of the portfolio per trade)
- records every decision so you can review it
- trades entirely with pretend money

## What you don't have yet

- **Automatic signals.** Right now you send them by hand. Real automatic
  alerts need a *paid* TradingView plan (webhooks aren't on the free tier).
- **24/7 running.** It only runs while your PC is on and the black window
  is open. Running it around the clock needs a rented server (~$5-10/month).
- **Real trading.** That's deliberately switched off, and should stay off
  for a long time. See below.

---

## Before you ever consider real money

Read this part properly.

1. **Paper trade for weeks, not hours.** The point is to find out whether
   the strategy actually makes money. It might not. Far better to learn
   that from a simulation.

2. **Never fund it with money you can't afford to lose completely.**
   Memecoins routinely go to zero. The scam filter blocks the obvious
   traps; it cannot make a bad bet good.

3. **Don't run it live until you can comfortably start, stop, and read
   it yourself.** A bot you can't stop is dangerous. If something goes
   wrong at 3am you need to be able to shut it off and understand what
   happened.

4. **A bot is not an edge.** It removes emotion and enforces discipline.
   It does not predict prices. Most people who trade memecoins lose
   money, and automating it doesn't change that.

The full technical documentation is in [README.md](README.md), and server
deployment is in [deploy/vps_setup.md](deploy/vps_setup.md), whenever
you're ready for them.
