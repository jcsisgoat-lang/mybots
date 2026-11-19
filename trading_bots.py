"""
4-bot paper trader for Alpaca (stocks + crypto), long-only, daily-bar strategies.

Bots (each owns its own symbols, so profit/loss can be tracked per bot):
  meanrev   stocks   RSI(2) dip-buying in an uptrend          (high win rate, small target)
  trend     stocks   pullback re-entry in an uptrend          (lower win rate, bigger winners)
  breakout  crypto   20-day high breakout                     (lower win rate, bigger winners)
  dip       crypto   RSI(5) oversold dip in an uptrend        (higher win rate, small target)

Safety:
  * PAPER = True. Nothing here touches real money.
  * Risk per trade, daily loss limit, per-bot pause, and a kill switch.

Use in Google Colab:
    1) !pip install -q alpaca-py
    2) %run trading_bots.py          (it asks for your key + secret)
    3) dashboard()                   see profit/loss per bot
       run_once()                    one pass: check signals, place/exit trades
       run_forever(15)               repeat every 15 minutes (stop with the stop button)
       kill()                        emergency stop: cancel orders, close everything, pause bots
       resume()                      remove the kill switch
"""
import os
import json
import math
import time
from datetime import datetime, timedelta, timezone
from getpass import getpass

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, CryptoBarsRequest, StockLatestTradeRequest, CryptoLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ------------------------------------------------------------------ settings
PAPER = True                 # keep True. Live trading is a deliberate, separate step.
RISK_PER_TRADE = 0.01        # risk 1% of a bot's allocation on each trade (distance to stop)
DAILY_LOSS_LIMIT = 0.02      # stop opening new trades if account is down 2% on the day
BOT_PAUSE_DRAWDOWN = 0.08    # pause a bot if it is down 8% of its allocation
STATE_FILE = "bot_state.json"
KILL_FILE = "KILL"


# ------------------------------------------------------------------ indicators
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd)


def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------ strategies
# entry(df) -> {"sl": ATR multiple for stop, "tp": ATR multiple for target} or None
# exit(df)  -> True when the position should be closed
def meanrev_entry(df):
    c = df.close
    if c.iloc[-1] > sma(c, 200).iloc[-1] and rsi(c, 2).iloc[-1] < 10:
        return {"sl": 3.0, "tp": 1.0}


def meanrev_exit(df):
    return df.close.iloc[-1] > sma(df.close, 5).iloc[-1]


def trend_entry(df):
    c = df.close
    uptrend = sma(c, 50).iloc[-1] > sma(c, 200).iloc[-1]
    e = ema(c, 20)
    if uptrend and c.iloc[-2] <= e.iloc[-2] and c.iloc[-1] > e.iloc[-1]:
        return {"sl": 2.5, "tp": 6.0}


def trend_exit(df):
    return df.close.iloc[-1] < sma(df.close, 50).iloc[-1]


def breakout_entry(df):
    c = df.close
    prior_high = df.high.iloc[-21:-1].max()
    if c.iloc[-1] > prior_high and c.iloc[-1] > sma(c, 100).iloc[-1]:
        return {"sl": 2.0, "tp": 4.0}


def breakout_exit(df):
    return df.close.iloc[-1] < df.low.iloc[-11:-1].min()


def dip_entry(df):
    c = df.close
    if c.iloc[-1] > sma(c, 100).iloc[-1] and rsi(c, 5).iloc[-1] < 25:
        return {"sl": 2.5, "tp": 2.0}


def dip_exit(df):
    return rsi(df.close, 5).iloc[-1] > 60


BOTS = [
    {"name": "meanrev",  "market": "stock",  "symbols": ["SPY", "QQQ", "IWM", "DIA"],
     "entry": meanrev_entry,  "exit": meanrev_exit,  "share": 0.25},
    {"name": "trend",    "market": "stock",  "symbols": ["AAPL", "MSFT", "NVDA", "AMZN"],
     "entry": trend_entry,    "exit": trend_exit,    "share": 0.25},
    {"name": "breakout", "market": "crypto", "symbols": ["BTC/USD", "ETH/USD"],
     "entry": breakout_entry, "exit": breakout_exit, "share": 0.25},
    {"name": "dip",      "market": "crypto", "symbols": ["SOL/USD", "LTC/USD"],
     "entry": dip_entry,      "exit": dip_exit,      "share": 0.25},
]


# ------------------------------------------------------------------ connection
def _keys():
    k = os.environ.get("ALPACA_API_KEY") or getpass("API key: ")
    s = os.environ.get("ALPACA_SECRET_KEY") or getpass("Secret key: ")
    return k.strip(), s.strip()


KEY, SECRET = _keys()
trading = TradingClient(KEY, SECRET, paper=PAPER)
stock_data = StockHistoricalDataClient(KEY, SECRET)
crypto_data = CryptoHistoricalDataClient(KEY, SECRET)


# ------------------------------------------------------------------ helpers
def norm(sym):
    return sym.replace("/", "")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"signals": {}, "levels": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def get_bars(bot, sym):
    """Completed daily bars only (today's unfinished bar is dropped)."""
    start = datetime.now(timezone.utc) - timedelta(days=400)
    if bot["market"] == "stock":
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                               start=start, feed=DataFeed.IEX)
        df = stock_data.get_stock_bars(req).df
    else:
        req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=start)
        df = crypto_data.get_crypto_bars(req).df
    df = df.droplevel(0)
    df.index = pd.to_datetime(df.index)
    return df[df.index.date < datetime.now(timezone.utc).date()]


def latest_price(bot, sym, fallback):
    try:
        if bot["market"] == "stock":
            t = stock_data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=sym, feed=DataFeed.IEX))
        else:
            t = crypto_data.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=sym))
        return float(t[sym].price)
    except Exception:
        return fallback


def open_orders():
    return trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))


def close_symbol(sym):
    key = norm(sym)
    for o in open_orders():
        if norm(o.symbol) == key:
            trading.cancel_order_by_id(o.id)
    time.sleep(1)
    trading.close_position(key)


# ------------------------------------------------------------------ orders
def place_entry(bot, sym, sig, df, alloc, state):
    key = norm(sym)
    last = float(df.close.iloc[-1])
    a = float(atr(df).iloc[-1])
    px = latest_price(bot, sym, last)
    sl = px - sig["sl"] * a
    tp = px + sig["tp"] * a
    stop_dist = px - sl
    if stop_dist <= 0 or px <= 0:
        return
    qty = min(alloc * RISK_PER_TRADE / stop_dist,      # risk-based size
              alloc / len(bot["symbols"]) / px)        # never more than this symbol's slice
    cid = f"{bot['name']}-{key}-{datetime.now(timezone.utc):%Y%m%d%H%M}"

    if bot["market"] == "stock":
        qty = math.floor(qty)
        if qty < 1:
            print(f"  {bot['name']}: {sym} signal but position size < 1 share, skipped")
            return
        req = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl, 2)),
            client_order_id=cid)
    else:
        qty = round(qty, 6)
        if qty <= 0 or qty * px < 10:
            print(f"  {bot['name']}: {sym} signal but order under $10, skipped")
            return
        req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.BUY,
                                 time_in_force=TimeInForce.GTC, client_order_id=cid)

    trading.submit_order(req)
    if bot["market"] == "crypto":      # crypto has no bracket orders: this bot watches the levels itself
        state["levels"][key] = {"sl": sl, "tp": tp}
    print(f"  {bot['name']}: BUY {qty} {sym} ~{px:.2f}  stop {sl:.2f}  target {tp:.2f}")


def handle_exit(bot, sym, df, state):
    key = norm(sym)
    px = latest_price(bot, sym, float(df.close.iloc[-1]))
    reason = "exit signal" if bot["exit"](df) else None
    if bot["market"] == "crypto":
        lv = state["levels"].get(key)
        if lv and px <= lv["sl"]:
            reason = "stop loss"
        elif lv and px >= lv["tp"]:
            reason = "take profit"
    if reason:
        try:
            close_symbol(sym)
            state["levels"].pop(key, None)
            print(f"  {bot['name']}: CLOSE {sym} ({reason}) ~{px:.2f}")
        except Exception as e:
            print(f"  {bot['name']}: could not close {sym} yet ({e}); will retry next pass")


# ------------------------------------------------------------------ stats / dashboard
def bot_stats():
    sym2bot = {norm(s): b["name"] for b in BOTS for s in b["symbols"]}
    stats = {b["name"]: {"realized": 0.0, "wins": 0, "losses": 0, "unrealized": 0.0, "open": []}
             for b in BOTS}

    orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, nested=True))
    fills = []
    for o in orders:
        for it in [o] + list(o.legs or []):
            q = float(it.filled_qty or 0)
            if q > 0 and it.filled_avg_price is not None and norm(it.symbol) in sym2bot:
                side = getattr(it.side, "value", it.side)
                ts = it.filled_at.timestamp() if it.filled_at else 0
                fills.append((ts, norm(it.symbol), side, q, float(it.filled_avg_price)))
    fills.sort(key=lambda f: f[0])

    book = {}   # symbol -> [qty, average cost]
    for _, key, side, q, p in fills:
        s = stats[sym2bot[key]]
        pq, pa = book.get(key, [0.0, 0.0])
        if side == "buy":
            nq = pq + q
            book[key] = [nq, (pq * pa + q * p) / nq]
        else:
            pnl = (p - pa) * min(q, pq)
            s["realized"] += pnl
            s["wins" if pnl > 0 else "losses"] += 1
            book[key] = [max(pq - q, 0.0), pa]

    for p in trading.get_all_positions():
        k = norm(p.symbol)
        if k in sym2bot:
            s = stats[sym2bot[k]]
            s["unrealized"] += float(p.unrealized_pl)
            s["open"].append(f"{k} x{float(p.qty):g}")
    return stats


def dashboard():
    acct = trading.get_account()
    eq, last = float(acct.equity), float(acct.last_equity)
    print(f"Account equity: ${eq:,.2f}   today: {eq - last:+,.2f}   ({'PAPER' if PAPER else 'LIVE'})\n")
    stats = bot_stats()
    rows = []
    for b in BOTS:
        s = stats[b["name"]]
        alloc = eq * b["share"]
        total = s["realized"] + s["unrealized"]
        n = s["wins"] + s["losses"]
        rows.append({
            "bot": b["name"], "market": b["market"],
            "allocation": f"${alloc:,.0f}",
            "total P/L": f"{total:+,.2f}",
            "return": f"{total / alloc * 100:+.2f}%",
            "closed trades": n,
            "win rate": f"{s['wins'] / n * 100:.0f}%" if n else "-",
            "open": ", ".join(s["open"]) or "-",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAlpaca's own dashboard shows the official balance: compare it against this.")


# ------------------------------------------------------------------ kill switch
def kill():
    open(KILL_FILE, "w").close()
    trading.cancel_orders()
    trading.close_all_positions(cancel_orders=True)
    print("KILL SWITCH ON: orders cancelled, positions closing, bots paused. Run resume() to restart.")


def resume():
    if os.path.exists(KILL_FILE):
        os.remove(KILL_FILE)
    print("Kill switch removed. Bots can trade again on the next run_once().")


# ------------------------------------------------------------------ main loop
def run_once():
    print(f"\n--- {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ---")
    if os.path.exists(KILL_FILE):
        print("Kill switch is ON (file 'KILL' exists). Doing nothing. Run resume() to restart.")
        return

    acct = trading.get_account()
    equity, last_equity = float(acct.equity), float(acct.last_equity)
    halted = equity < last_equity * (1 - DAILY_LOSS_LIMIT)
    if halted:
        print(f"Daily loss limit hit ({equity - last_equity:+,.2f} today): no new trades.")

    market_open = trading.get_clock().is_open
    positions = {norm(p.symbol): p for p in trading.get_all_positions()}
    busy = {norm(o.symbol) for o in open_orders()}
    state = load_state()
    stats = bot_stats()

    for bot in BOTS:
        if bot["market"] == "stock" and not market_open:
            continue                                    # stock bots only act while the market is open
        alloc = equity * bot["share"]
        s = stats[bot["name"]]
        paused = (s["realized"] + s["unrealized"]) < -BOT_PAUSE_DRAWDOWN * alloc
        if paused:
            print(f"  {bot['name']}: paused (down more than {BOT_PAUSE_DRAWDOWN:.0%} of its allocation)")

        for sym in bot["symbols"]:
            key = norm(sym)
            try:
                df = get_bars(bot, sym)
            except Exception as e:
                print(f"  {bot['name']}: no data for {sym} ({e})")
                continue
            if len(df) < 210:
                print(f"  {bot['name']}: not enough history for {sym}")
                continue

            if key in positions:
                handle_exit(bot, sym, df, state)
                continue
            if key in busy or halted or paused:
                continue

            sig = bot["entry"](df)
            bar_date = str(df.index[-1].date())
            if sig and state["signals"].get(key) != bar_date:   # one entry per signal bar
                try:
                    place_entry(bot, sym, sig, df, alloc, state)
                    state["signals"][key] = bar_date
                except Exception as e:
                    print(f"  {bot['name']}: order for {sym} failed ({e})")

    save_state(state)
    print("Pass complete.")


def run_forever(minutes=15):
    print("Running. Press the stop button in Colab to end it.")
    while True:
        try:
            run_once()
        except Exception as e:
            print("Error this pass:", e)
        time.sleep(minutes * 60)
