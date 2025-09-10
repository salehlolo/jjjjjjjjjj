#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX Demo Scalper — Top-N Scanner (x10, +1% TP)
-----------------------------------------------
Adds scanning the top-N (default 40) USDT-margined perpetuals by 24h quote volume.
- One position at a time (global), per your requirement.
- When flat: scan list -> pick first symbol with signal -> trade it.
- When in trade: manage only that symbol until exit, then scan again.

Base strategy/features:
- EMA200 regime, VWAP cross + EMA20 slope confirmation
- OCO-style exits (native if available; else client-side)
- Persistence & resume (state JSON)
- CSV logs (trades/daily/hourly) + Telegram alerts
- Fixed margin per trade (e.g., 90 USDT on x10) or risk% sizing

Env adds (besides previous):
- ENABLE_TOPN_SCAN=true
- TOPN=40
- QUOTE_FILTER=USDT
"""
import os, time, math, csv, json, traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------- CONFIG ----------------------
ENABLE_TOPN_SCAN = os.getenv("ENABLE_TOPN_SCAN", "true").lower() == "true"
TOPN = int(os.getenv("TOPN", "40"))
QUOTE_FILTER = os.getenv("QUOTE_FILTER", "USDT")

SYMBOL = os.getenv("OKX_SYMBOL", "BTC/USDT:USDT")  # fallback / default
TIMEFRAME = os.getenv("OKX_TIMEFRAME", "5m")
LEVERAGE = int(os.getenv("OKX_LEVERAGE", "10"))
RISK_PCT = float(os.getenv("OKX_RISK_PCT", "0.01"))
FIXED_MARGIN_USDT = float(os.getenv("FIXED_MARGIN_USDT", "90"))  # default to 90 for your request

ENTRY_TYPE = os.getenv("ENTRY_TYPE", "limit").lower()
MAKER_ENTRY = os.getenv("OKX_MAKER_ENTRY", "true").lower() == "true"
ISOLATED = os.getenv("OKX_ISOLATED", "true").lower() == "true"
HEDGED = os.getenv("OKX_HEDGED", "false").lower() == "true"

USE_NATIVE_BRACKET = os.getenv("USE_NATIVE_BRACKET", "true").lower() == "true"
USE_PARTIAL = os.getenv("USE_PARTIAL", "true").lower() == "true"

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
VWAP_NEAR_PCT = float(os.getenv("VWAP_NEAR_PCT", "0.001"))
ENTRY_OFFSET_PCT = float(os.getenv("ENTRY_OFFSET_PCT", "0.0002"))

TP_MAIN_PCT = float(os.getenv("TP_MAIN_PCT", "0.01"))
TP_PARTIAL_PCT = float(os.getenv("TP_PARTIAL_PCT", "0.007"))
PARTIAL_RATIO = float(os.getenv("PARTIAL_RATIO", "0.3"))

SL_MIN_PCT = float(os.getenv("SL_MIN_PCT", "0.005"))
SL_MAX_PCT = float(os.getenv("SL_MAX_PCT", "0.008"))
ATR_MULT = float(os.getenv("ATR_MULT", "0.75"))

POLL_SEC = int(os.getenv("POLL_SEC", "10"))
CANCEL_ENTRY_SEC = int(os.getenv("CANCEL_ENTRY_SEC", "600"))

COOLDOWN_MIN_AFTER_SL = int(os.getenv("COOLDOWN_MIN_AFTER_SL", "30"))
DIRECTIONAL_LOCK_LOSSES = int(os.getenv("DIRECTIONAL_LOCK_LOSSES", "3"))
DIRECTIONAL_LOCK_MIN = int(os.getenv("DIRECTIONAL_LOCK_MIN", "45"))
DAILY_MAX_LOSS_USDT = float(os.getenv("DAILY_MAX_LOSS_USDT", "0"))

FEE_BPS_TAKER = float(os.getenv("FEE_BPS_TAKER", "6"))
FEE_BPS_MAKER = float(os.getenv("FEE_BPS_MAKER", "2"))

STATE_PATH = os.getenv("STATE_PATH", "okx_state.json")
LOG_PATH = os.getenv("LOG_PATH", "trades_log.csv")
DAILY_CSV = os.getenv("DAILY_CSV", "daily_performance.csv")
HOURLY_CSV = os.getenv("HOURLY_CSV", "hourly_report.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "")
API_PASSWORD = os.getenv("OKX_PASSWORD", "")

# ---------------------- UTILS ----------------------
def utcnow(): return datetime.now(tz=timezone.utc)
def now_ms(): return int(utcnow().timestamp() * 1000)

def ema(s: pd.Series, span: int) -> pd.Series: return s.ewm(span=span, adjust=False).mean()
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def compute_vwap_intraday(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    day_key = ts.floor('D')
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'].replace(0, np.nan).fillna(method='ffill')
    cum_pv = (tp * vol).groupby(day_key).cumsum()
    cum_v = vol.groupby(day_key).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def clamp(x, lo, hi): return max(lo, min(hi, x))

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[telegram] failed: {e}")

def fmt_usd(x):
    try: return f"${float(x):,.2f}"
    except: return str(x)

def ensure_csv_headers(path: str, headers: list):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

# ---------------------- PERSISTENT STATE ----------------------
class Store:
    def __init__(self, path: str):
        self.path = path
        self.data = {
            "equity_start": None, "cum_pnl_usdt": 0.0,
            "trades": 0, "wins": 0, "losses": 0,
            "loss_streak_long": 0, "loss_streak_short": 0,
            "cooldown_until": 0, "lock_until": {"long":0,"short":0},
            "last_hourly_report_ms": 0, "current_day": None,
            "daily_net_usdt": 0.0, "daily_trades": 0, "daily_wins": 0, "daily_losses": 0,
            "pos": {"symbol": None, "side": None, "entry_price": None, "size": 0.0,
                    "tp_partial": None, "tp_main": None, "sl": None, "partial_done": False,
                    "entry_order_id": None, "tp_partial_id": None, "tp_main_id": None, "sl_id": None,
                    "entry_time": 0}
        }
        self.load()
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: self.data.update(json.load(f))
                print(f"[*] Loaded state from {self.path}")
            except Exception as e: print(f"[warn] failed to load state: {e}")
    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f: json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e: print(f"[warn] failed to save state: {e}")

# ---------------------- EXCHANGE & MARKETS ----------------------
def build_exchange() -> ccxt.Exchange:
    ex = ccxt.okx({"apiKey": API_KEY, "secret": API_SECRET, "password": API_PASSWORD,
                   "enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.set_sandbox_mode(True); ex.load_markets()
    try: ex.set_position_mode(HEDGED)
    except Exception: print("[warn] set_position_mode not supported; continuing in default mode.")
    try: ex.set_leverage(LEVERAGE, SYMBOL, {"mgnMode": "isolated" if ISOLATED else "cross"})
    except Exception as e: print(f"[warn] set_leverage failed: {e}")
    return ex

def topn_symbols_okx(ex: ccxt.Exchange, topn: int = TOPN, quote: str = QUOTE_FILTER) -> List[str]:
    # Filter swap (perpetual) markets quoted in 'quote'
    symbols = []
    for s, m in ex.markets.items():
        if m.get('type') == 'swap' and m.get('quote') == quote and m.get('active', True):
            symbols.append(m['symbol'])
    # Score by 24h quote volume
    try:
        tickers = ex.fetch_tickers(symbols)
        scored = []
        for sym, t in tickers.items():
            v = 0.0
            info = t.get('info', {})
            for k in ('volCcy24h', 'quoteVolume', 'vol24h'):
                if k in info and info[k]:
                    try: v = float(info[k]); break
                    except: pass
            if v == 0.0:
                try: v = float(t.get('quoteVolume') or 0.0)
                except: v = 0.0
            scored.append((v, sym))
        scored.sort(reverse=True)
        return [sym for v, sym in scored[:topn]]
    except Exception as e:
        print(f"[warn] fetch_tickers failed: {e}; fallback single symbol {SYMBOL}")
        return [SYMBOL]

def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])

def get_equity_usdt(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance(); usdt = bal.get('USDT') or {}
    total = usdt.get('total'); 
    if total is None: total = (usdt.get('free',0) or 0)+(usdt.get('used',0) or 0)
    return float(total or 0)

def current_price(ex: ccxt.Exchange, symbol: str) -> float:
    return float(ex.fetch_ticker(symbol)['last'])

def has_open_position(ex: ccxt.Exchange, symbol: str) -> Tuple[bool, Optional[str], float]:
    try:
        poss = ex.fetch_positions([symbol])
        for p in poss:
            amt = float(p.get('contracts') or p.get('positionAmt') or p.get('info', {}).get('pos', 0) or 0)
            if abs(amt) > 0: return True, ('long' if amt>0 else 'short'), amt
    except Exception as e: print(f"[warn] fetch_positions failed: {e}")
    return False, None, 0.0

# ---------------------- STRATEGY ----------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ema_fast'] = ema(out['close'], EMA_FAST)
    out['ema_slow'] = ema(out['close'], EMA_SLOW)
    out['atr'] = atr(out, ATR_LEN)
    out['atr_pct'] = out['atr'] / out['close']
    out['vwap'] = compute_vwap_intraday(out)
    return out

def entry_signal(row_prev, row):
    long_cross = (row_prev['close'] < row_prev['vwap']) and (row['close'] > row['vwap'])
    short_cross = (row_prev['close'] > row_prev['vwap']) and (row['close'] < row['vwap'])
    ema20_up = row['ema_fast'] > row_prev['ema_fast']
    ema20_dn = row['ema_fast'] < row_prev['ema_fast']
    if long_cross and (row['close'] > row['ema_slow']) and ema20_up: return "long"
    if short_cross and (row['close'] < row['ema_slow']) and ema20_dn: return "short"
    return None

def compute_sl_tp(entry_px: float, atr_pct: float, side: str):
    sl_dist = clamp(max(SL_MIN_PCT, ATR_MULT * atr_pct), SL_MIN_PCT, SL_MAX_PCT)
    if side == "long":
        return entry_px*(1-sl_dist), entry_px*(1+TP_PARTIAL_PCT), entry_px*(1+TP_MAIN_PCT)
    else:
        return entry_px*(1+sl_dist), entry_px*(1-TP_PARTIAL_PCT), entry_px*(1-TP_MAIN_PCT)

def size_from_risk(ex: ccxt.Exchange, symbol: str, entry_px: float, sl_px: float) -> float:
    if FIXED_MARGIN_USDT > 0: notional = FIXED_MARGIN_USDT * LEVERAGE
    else:
        equity = get_equity_usdt(ex); stop_pct = abs((entry_px - sl_px) / entry_px)
        if stop_pct <= 0: return 0.0
        notional = (equity * RISK_PCT) / stop_pct
    amount = notional / entry_px
    amt_prec = ex.markets[symbol]['precision'].get('amount', 6)
    amount = float(f"{amount:.{amt_prec}f}")
    min_amt = ex.markets[symbol]['limits']['amount'].get('min') or 0
    return max(amount, min_amt)

# ---------------------- ORDER HELPERS ----------------------
def place_entry(ex: ccxt.Exchange, symbol: str, side: str, amount: float, last_px: float):
    order_side = "buy" if side == "long" else "sell"
    params = {"tdMode": "isolated" if ISOLATED else "cross", "reduceOnly": False, "lever": str(LEVERAGE)}
    if ENTRY_TYPE == "limit":
        price = last_px * (1 - ENTRY_OFFSET_PCT) if side=="long" else last_px * (1 + ENTRY_OFFSET_PCT)
        price = float(ex.price_to_precision(symbol, price))
        params["postOnly"] = True if MAKER_ENTRY else False
        o = ex.create_order(symbol, "limit", order_side, amount, price, params)
        return o['id'], price
    else:
        o = ex.create_order(symbol, "market", order_side, amount, None, params)
        avg = float(o.get('average') or last_px); return o['id'], avg

def try_place_native_brackets(ex, symbol, side, amount, sl, tp_partial, tp_main, partial_ratio):
    params = {"tdMode": "isolated" if ISOLATED else "cross", "reduceOnly": True}
    ids = {"tp_partial_id": None, "tp_main_id": None, "sl_id": None}
    try:
        qty_p = float(ex.amount_to_precision(symbol, amount * partial_ratio))
        if qty_p > 0 and USE_PARTIAL:
            p = float(ex.price_to_precision(symbol, tp_partial))
            o1 = ex.create_order(symbol, "limit", "sell" if side=="long" else "buy", qty_p, p, params); ids["tp_partial_id"] = o1['id']
    except Exception: pass
    try:
        qty_m = float(ex.amount_to_precision(symbol, amount * (1-(partial_ratio if USE_PARTIAL else 0.0))))
        if qty_m > 0:
            p = float(ex.price_to_precision(symbol, tp_main))
            o2 = ex.create_order(symbol, "limit", "sell" if side=="long" else "buy", qty_m, p, params); ids["tp_main_id"] = o2['id']
    except Exception: pass
    try:
        sl_params = params.copy(); sl_params["stopLossPrice"] = float(ex.price_to_precision(symbol, sl))
        o3 = ex.create_order(symbol, "market", "sell" if side=="long" else "buy", float(ex.amount_to_precision(symbol, amount)), None, sl_params)
        ids["sl_id"] = o3['id']
    except Exception: pass
    return ids

def cancel_order_safe(ex, symbol, order_id): 
    if not order_id: return
    try: ex.cancel_order(order_id, symbol)
    except Exception: pass

def ensure_exit(ex, symbol, side, amount):
    params = {"tdMode": "isolated" if ISOLATED else "cross", "reduceOnly": True}
    ex.create_order(symbol, "market", ("sell" if side=="long" else "buy"), amount, None, params)

# ---------------------- REPORTING ----------------------
def trades_csv_headers(): return ["time","symbol","event","side","price","size","entry_price","sl","tp_partial","tp_main","pnl_usdt","fees_est_usdt","info"]
def daily_csv_headers(): return ["date","trades","wins","losses","gross_profit_usdt","gross_loss_usdt","net_usdt","net_pct","fees_est_usdt"]
def hourly_csv_headers(): return ["time","trades_total","wins","losses","net_usdt","net_pct","best_trade_usdt","worst_trade_usdt","fees_est_usdt"]

def estimate_fees_usdt(entry_price, exit_price, amount, maker=True):
    bps = FEE_BPS_MAKER if maker else FEE_BPS_TAKER
    return (bps/10000.0) * ((entry_price*amount) + (exit_price*amount))

def append_trade_log(path, **row):
    ensure_csv_headers(path, trades_csv_headers())
    row.setdefault("time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=trades_csv_headers()).writerow(row)

def update_daily_csv(date_str, trades, wins, losses, gross_profit, gross_loss, net, net_pct, fees):
    ensure_csv_headers(DAILY_CSV, daily_csv_headers())
    with open(DAILY_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=daily_csv_headers()).writerow({
            "date": date_str, "trades": trades, "wins": wins, "losses": losses,
            "gross_profit_usdt": f"{gross_profit:.4f}", "gross_loss_usdt": f"{gross_loss:.4f}",
            "net_usdt": f"{net:.4f}", "net_pct": f"{net_pct:.4f}", "fees_est_usdt": f"{fees:.4f}",
        })

def append_hourly_csv(trades, wins, losses, net, net_pct, best, worst, fees):
    ensure_csv_headers(HOURLY_CSV, hourly_csv_headers())
    with open(HOURLY_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=hourly_csv_headers()).writerow({
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "trades_total": trades, "wins": wins, "losses": losses, "net_usdt": f"{net:.4f}",
            "net_pct": f"{net_pct:.4f}", "best_trade_usdt": f"{best:.4f}", "worst_trade_usdt": f"{worst:.4f}",
            "fees_est_usdt": f"{fees:.4f}"
        })

def hourly_report_telegram(store):
    eq0 = store.data.get("equity_start") or 0.0
    net = float(store.data.get("cum_pnl_usdt") or 0.0)
    trades = store.data.get("trades") or 0
    wins = store.data.get("wins") or 0
    losses = store.data.get("losses") or 0
    net_pct = (net/eq0*100) if eq0>0 else 0.0
    best = worst = fees = 0.0
    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            if "pnl_usdt" in df.columns:
                vals = pd.to_numeric(df["pnl_usdt"], errors="coerce").dropna()
                if len(vals)>0: best, worst = float(vals.max()), float(vals.min())
            if "fees_est_usdt" in df.columns:
                fees = float(pd.to_numeric(df["fees_est_usdt"], errors="coerce").fillna(0).sum())
    except Exception: pass
    append_hourly_csv(trades, wins, losses, net, net_pct, best, worst, fees)
    send_telegram(f"⏰ <b>تقرير الساعة</b>\nالصفقات: {trades} | ربح: {wins} | خسارة: {losses}\nصافي: <b>{fmt_usd(net)}</b> ({net_pct:.2f}%)\nأفضل: {fmt_usd(best)} | أسوأ: {fmt_usd(worst)}\nرسوم: {fmt_usd(fees)}")

# ---------------------- MAIN ----------------------
def run_loop():
    ex = build_exchange()
    store = Store(STATE_PATH)
    if not store.data.get("equity_start"): store.data["equity_start"] = get_equity_usdt(ex); store.save()
    today = utcnow().date().isoformat()
    if store.data.get("current_day") != today:
        store.data.update({"current_day":today, "daily_net_usdt":0.0, "daily_trades":0, "daily_wins":0, "daily_losses":0}); store.save()
    print(f"[*] OKX DEMO | TF={TIMEFRAME} | lev x{LEVERAGE} | fixed_margin={FIXED_MARGIN_USDT} | scan={ENABLE_TOPN_SCAN} TOPN={TOPN}")
    last_hour = utcnow().replace(minute=0, second=0, microsecond=0)

    # If exchange has no open pos, clear persisted pos
    in_pos, pos_side, pos_amt = has_open_position(ex, store.data['pos']['symbol'] or SYMBOL)
    if not in_pos:
        store.data["pos"].update({"symbol":None,"side":None,"entry_price":None,"size":0.0,"tp_partial":None,"tp_main":None,"sl":None,
                                  "partial_done":False,"entry_order_id":None,"tp_partial_id":None,"tp_main_id":None,"sl_id":None,"entry_time":0})
        store.save()

    symbols_scan = [SYMBOL]
    last_scan = 0

    while True:
        try:
            # Hourly report
            if utcnow() - last_hour >= timedelta(hours=1):
                hourly_report_telegram(store)
                last_hour = utcnow().replace(minute=0, second=0, microsecond=0)

            # Daily stop
            pass_allowed = not (DAILY_MAX_LOSS_USDT>0 and store.data["daily_net_usdt"] <= -abs(DAILY_MAX_LOSS_USDT))
            if now_ms() < (store.data.get("cooldown_until") or 0): pass_allowed = False

            # If flat: scan watchlist (refresh every 5 minutes)
            pos_symbol = store.data['pos']['symbol'] or SYMBOL
            in_pos, pos_side, pos_amt = has_open_position(ex, pos_symbol)
            if (not in_pos) and pass_allowed and (store.data["pos"]["entry_order_id"] is None):
                if ENABLE_TOPN_SCAN and (now_ms() - last_scan > 5*60*1000 or len(symbols_scan)==1):
                    symbols_scan = topn_symbols_okx(ex, TOPN, QUOTE_FILTER); last_scan = now_ms()

                selected = None; side = None; row = row_prev = None; last_px = None
                for sym in symbols_scan:
                    try:
                        df = fetch_ohlcv_df(ex, sym, TIMEFRAME, limit=200)
                        ind = compute_indicators(df).dropna()
                        r_prev, r = ind.iloc[-2], ind.iloc[-1]
                        s = entry_signal(r_prev, r)
                        # lockout on direction
                        if s:
                            lk = store.data.get("lock_until", {"long":0,"short":0})
                            if now_ms() >= (lk.get(s,0) or 0):
                                selected, side, row_prev, row = sym, s, r_prev, r
                                last_px = float(r['close'])
                                break
                    except Exception as e:
                        continue

                if selected and side:
                    # Switch active symbol
                    store.data['pos']['symbol'] = selected; store.save()
                    sl, tp_partial, tp_main = compute_sl_tp(last_px, float(row['atr_pct']), side)
                    amount = size_from_risk(ex, selected, last_px, sl)
                    if amount > 0:
                        order_id, entry_price = place_entry(ex, selected, side, amount, last_px)
                        store.data["pos"].update({"side":side, "size":amount, "entry_price":entry_price, "tp_partial":tp_partial,
                                                  "tp_main":tp_main, "sl":sl, "partial_done":False, "entry_order_id":order_id, "entry_time":now_ms()})
                        store.save()
                        append_trade_log(LOG_PATH, event="plan", symbol=selected, side=side, price=last_px, size=amount,
                                         entry_price=entry_price, sl=sl, tp_partial=tp_partial, tp_main=tp_main,
                                         pnl_usdt="", fees_est_usdt="", info="scan")
                        send_telegram(f"🎯 <b>خطة دخول</b> {selected} {side.upper()} @ {entry_price:.2f}\nSL {sl:.2f} | TPp {tp_partial:.2f} | TP {tp_main:.2f}")

            # Manage pending entry
            pos_symbol = store.data['pos']['symbol'] or SYMBOL
            if store.data["pos"]["entry_order_id"] and not has_open_position(ex, pos_symbol)[0]:
                try:
                    o = ex.fetch_order(store.data["pos"]["entry_order_id"], pos_symbol); status = (o.get('status') or '').lower()
                    if status in ("closed","filled"):
                        filled = float(o.get('filled') or 0); avg = float(o.get('average') or store.data["pos"]["entry_price"] or current_price(ex, pos_symbol))
                        if filled>0: store.data["pos"]["entry_price"] = avg
                        if USE_NATIVE_BRACKET:
                            ids = try_place_native_brackets(ex, pos_symbol, store.data["pos"]["side"], store.data["pos"]["size"],
                                                            store.data["pos"]["sl"], store.data["pos"]["tp_partial"], store.data["pos"]["tp_main"], PARTIAL_RATIO)
                            store.data["pos"].update(ids)
                        send_telegram(f"✅ <b>تم الدخول</b> {pos_symbol} {store.data['pos']['side'].upper()} @ {store.data['pos']['entry_price']:.2f}")
                        append_trade_log(LOG_PATH, event="entry", symbol=pos_symbol, side=store.data["pos"]["side"],
                                         price=store.data["pos"]["entry_price"], size=filled, entry_price=store.data["pos"]["entry_price"],
                                         sl=store.data["pos"]["sl"], tp_partial=store.data["pos"]["tp_partial"], tp_main=store.data["pos"]["tp_main"],
                                         pnl_usdt="", fees_est_usdt="", info=f"order={store.data['pos']['entry_order_id']}")
                        store.data["pos"]["entry_order_id"] = None; store.save()
                    elif status in ("canceled","cancelled"):
                        append_trade_log(LOG_PATH, event="entry-canceled", symbol=pos_symbol, side=store.data["pos"]["side"], price=current_price(ex, pos_symbol),
                                         size=0, entry_price="", sl="", tp_partial="", tp_main="", pnl_usdt="", fees_est_usdt="", info="")
                        send_telegram("ℹ️ تم إلغاء أمر الدخول من المنصة.")
                        store.data["pos"].update({"symbol":None,"side":None,"size":0.0,"entry_price":None,"tp_partial":None,"tp_main":None,"sl":None,
                                                  "partial_done":False,"entry_order_id":None,"tp_partial_id":None,"tp_main_id":None,"sl_id":None,"entry_time":0})
                        store.save()
                    else:
                        if now_ms() - (store.data["pos"]["entry_time"] or 0) > CANCEL_ENTRY_SEC*1000:
                            cancel_order_safe(ex, pos_symbol, store.data["pos"]["entry_order_id"])
                            append_trade_log(LOG_PATH, event="entry-stale-cancel", symbol=pos_symbol, side=store.data["pos"]["side"],
                                             price=current_price(ex, pos_symbol), size=0, entry_price="", sl="", tp_partial="", tp_main="",
                                             pnl_usdt="", fees_est_usdt="", info="ttl")
                            store.data["pos"].update({"symbol":None,"side":None,"size":0.0,"entry_price":None,"tp_partial":None,"tp_main":None,"sl":None,
                                                      "partial_done":False,"entry_order_id":None,"tp_partial_id":None,"tp_main_id":None,"sl_id":None,"entry_time":0})
                            store.save()
                except Exception as e: print(f"[warn] fetch_order failed: {e}")

            # Manage open position (client-side exits as fallback)
            pos_symbol = store.data['pos']['symbol'] or SYMBOL
            in_pos, pos_side, pos_amt = has_open_position(ex, pos_symbol)
            if in_pos:
                last_px = current_price(ex, pos_symbol); pos = store.data["pos"]
                if USE_PARTIAL and (not pos["partial_done"]) and (pos["tp_partial_id"] is None):
                    if (pos_side=="long" and last_px >= pos["tp_partial"]) or (pos_side=="short" and last_px <= pos["tp_partial"]):
                        qty = float(ex.amount_to_precision(pos_symbol, pos_amt * PARTIAL_RATIO))
                        if qty>0:
                            ensure_exit(ex, pos_symbol, pos_side, qty); pos["partial_done"] = True; pos["sl"] = pos["entry_price"]
                            append_trade_log(LOG_PATH, event="tp-partial", symbol=pos_symbol, side=pos_side, price=last_px, size=qty,
                                             entry_price=pos["entry_price"], sl=pos["sl"], tp_partial=pos["tp_partial"], tp_main=pos["tp_main"],
                                             pnl_usdt="", fees_est_usdt="", info="client")
                            send_telegram(f"🏁 {pos_symbol} جني جزئي {qty} @ {last_px:.2f} | SL → BE")

                should_tp = (pos_side=="long" and last_px >= pos["tp_main"]) or (pos_side=="short" and last_px <= pos["tp_main"])
                should_sl = (pos_side=="long" and last_px <= pos["sl"]) or (pos_side=="short" and last_px >= pos["sl"])
                if should_tp or should_sl:
                    qty = float(ex.amount_to_precision(pos_symbol, pos_amt))
                    if qty>0:
                        ensure_exit(ex, pos_symbol, pos_side, qty)
                        pnl = (last_px - pos["entry_price"]) * qty if pos_side=="long" else (pos["entry_price"] - last_px) * qty
                        fees = estimate_fees_usdt(pos["entry_price"], last_px, qty, maker=(ENTRY_TYPE=="limit" and MAKER_ENTRY))
                        net = pnl - fees
                        # stats
                        store.data["cum_pnl_usdt"] += net; store.data["trades"] += 1
                        is_win = net >= 0; store.data["wins"] += (1 if is_win else 0); store.data["losses"] += (0 if is_win else 1)
                        today = utcnow().date().isoformat()
                        if store.data.get("current_day") != today:
                            store.data.update({"current_day":today, "daily_net_usdt":0.0, "daily_trades":0, "daily_wins":0, "daily_losses":0})
                        store.data["daily_net_usdt"] += net; store.data["daily_trades"] += 1
                        if is_win: store.data["daily_wins"] += 1
                        else:
                            store.data["daily_losses"] += 1
                            store.data["cooldown_until"] = now_ms() + COOLDOWN_MIN_AFTER_SL*60*1000
                            key = "loss_streak_long" if pos_side=="long" else "loss_streak_short"
                            store.data[key] = (store.data.get(key) or 0) + 1
                            if store.data[key] >= DIRECTIONAL_LOCK_LOSSES:
                                store.data["lock_until"][pos_side] = now_ms() + DIRECTIONAL_LOCK_MIN*60*1000
                                store.data[key] = 0
                        # logs
                        append_trade_log(LOG_PATH, event=("tp" if should_tp else "sl"), symbol=pos_symbol, side=pos_side, price=last_px, size=qty,
                                         entry_price=pos["entry_price"], sl=pos["sl"], tp_partial=pos["tp_partial"], tp_main=pos["tp_main"],
                                         pnl_usdt=f"{net:.6f}", fees_est_usdt=f"{fees:.6f}", info="client")
                        eq0 = store.data.get("equity_start") or 0.0; total_pct = (store.data["cum_pnl_usdt"]/(eq0 or 1))*100
                        gross_profit = net if net>=0 else 0.0; gross_loss = -net if net<0 else 0.0
                        update_daily_csv(store.data["current_day"], store.data["daily_trades"], store.data["daily_wins"],
                                         store.data["daily_losses"], gross_profit, gross_loss, store.data["daily_net_usdt"],
                                         (store.data["daily_net_usdt"]/(eq0 or 1))*100, fees)
                        send_telegram(f"{'✅ ربح' if net>=0 else '❌ خسارة'} {pos_symbol}\n{pos_side.upper()} @ {pos['entry_price']:.2f} → {last_px:.2f}\nP&L: <b>{fmt_usd(net)}</b>")
                        # cleanup
                        cancel_order_safe(ex, pos_symbol, pos["tp_partial_id"]); cancel_order_safe(ex, pos_symbol, pos["tp_main_id"]); cancel_order_safe(ex, pos_symbol, pos["sl_id"])
                        store.data["pos"].update({"symbol":None,"side":None,"entry_price":None,"size":0.0,"tp_partial":None,"tp_main":None,"sl":None,
                                                  "partial_done":False,"entry_order_id":None,"tp_partial_id":None,"tp_main_id":None,"sl_id":None,"entry_time":0})
                        store.save()

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            print("\n[exit] stopped by user."); break
        except Exception as e:
            print("[error]", e); traceback.print_exc(); time.sleep(POLL_SEC)

if __name__ == "__main__":
    print("Starting OKX Demo Scalper (Top-N scanner)...")
    run_loop()
