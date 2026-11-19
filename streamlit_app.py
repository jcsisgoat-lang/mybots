"""
Phone-friendly dashboard for the trading bots (password protected).

Read-only website version for Streamlit Community Cloud.
Needs three secrets (set in the app's Secrets settings): ALPACA_API_KEY, ALPACA_SECRET_KEY, APP_PASSWORD
"""
import os
import hmac
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus

st.set_page_config(page_title="My Trading Bots", page_icon="📈", layout="centered")

# ---------------------------------------------------------------- password gate
PASSWORD = str(st.secrets.get("APP_PASSWORD", ""))
if not PASSWORD:
    st.error("APP_PASSWORD is not set in the app secrets.")
    st.stop()

if not st.session_state.get("authed"):
    st.title("My Trading Bots")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

os.environ["ALPACA_API_KEY"] = str(st.secrets["ALPACA_API_KEY"])
os.environ["ALPACA_SECRET_KEY"] = str(st.secrets["ALPACA_SECRET_KEY"])

import trading_bots as tb  # noqa: E402


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


@st.cache_data(ttl=60)
def load():
    acct = tb.trading.get_account()
    equity = float(acct.equity)
    today = equity - float(acct.last_equity)
    stats = tb.bot_stats()

    bots, total_all = [], 0.0
    for b in tb.BOTS:
        s = stats[b["name"]]
        alloc = equity * b["share"]
        total = s["realized"] + s["unrealized"]
        total_all += total
        n = s["wins"] + s["losses"]
        bots.append({
            "Bot": b["name"],
            "Market": b["market"],
            "Total P/L": money(total),
            "Return": f"{total / alloc * 100:+.2f}%",
            "Realized": money(s["realized"]),
            "Open P/L": money(s["unrealized"]),
            "Trades": n,
            "Win rate": f"{s['wins'] / n * 100:.0f}%" if n else "-",
            "Open positions": ", ".join(s["open"]) or "-",
        })

    # recent fills, newest first
    sym2bot = {tb.norm(sym): b["name"] for b in tb.BOTS for sym in b["symbols"]}
    orders = tb.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, nested=True))
    fills = []
    for o in orders:
        for it in [o] + list(o.legs or []):
            q = float(it.filled_qty or 0)
            if q > 0 and it.filled_avg_price is not None and it.filled_at and tb.norm(it.symbol) in sym2bot:
                fills.append((it.filled_at, {
                    "Time": it.filled_at.strftime("%b %d %H:%M"),
                    "Bot": sym2bot[tb.norm(it.symbol)],
                    "Symbol": it.symbol,
                    "Side": str(getattr(it.side, "value", it.side)).upper(),
                    "Qty": f"{q:g}",
                    "Price": f"${float(it.filled_avg_price):,.2f}",
                }))
    fills.sort(key=lambda x: x[0], reverse=True)

    curve = None
    try:
        ph = tb.trading.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
        curve = pd.Series(ph.equity, index=pd.to_datetime(ph.timestamp, unit="s")).dropna()
    except Exception:
        pass

    return {
        "equity": equity, "today": today, "total": total_all, "bots": bots,
        "recent": [f[1] for f in fills[:15]], "curve": curve,
        "stamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ---------------------------------------------------------------- page
d = load()

st.title("📈 My Trading Bots")
st.caption(f"{'PAPER' if tb.PAPER else 'LIVE'} account · updated {d['stamp']}")

c1, c2, c3 = st.columns(3)
c1.metric("Account equity", f"${d['equity']:,.2f}")
c2.metric("Today", money(d["today"]))
c3.metric("Made by bots", money(d["total"]))

if d["curve"] is not None and len(d["curve"]) > 1:
    st.line_chart(d["curve"])

st.subheader("Bots")
st.dataframe(pd.DataFrame(d["bots"]), hide_index=True)

st.subheader("Recent trades")
if d["recent"]:
    st.dataframe(pd.DataFrame(d["recent"]), hide_index=True)
else:
    st.info("No trades yet. The bots only trade when their conditions are met.")

if st.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()
