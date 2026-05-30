"""Technical analysis: risk metrics, candlestick patterns, seasonality."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk_metrics(df: pd.DataFrame, rf_daily: float = 0.000265) -> dict:
    """Sharpe, Sortino, VaR, CVaR, Max Drawdown, Calmar, Beta placeholder."""
    close   = df["Close"]
    returns = close.pct_change().dropna()
    ann     = 252

    sharpe  = (returns.mean() - rf_daily) / returns.std() * np.sqrt(ann)
    neg     = returns[returns < 0]
    sortino = (returns.mean() - rf_daily) / neg.std() * np.sqrt(ann) if len(neg) else np.nan
    var95   = float(np.percentile(returns, 5))
    cvar95  = float(returns[returns <= var95].mean())

    roll_max = close.cummax()
    dd       = (close - roll_max) / roll_max
    max_dd   = float(dd.min())

    total_ret = (close.iloc[-1] / close.iloc[0]) - 1
    years     = len(close) / ann
    calmar    = (total_ret / years) / abs(max_dd) if max_dd else np.nan

    ann_vol   = returns.std() * np.sqrt(ann)
    ann_ret   = returns.mean() * ann

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if not np.isnan(sortino) else np.nan,
        "var_95": round(var95 * 100, 2),
        "cvar_95": round(cvar95 * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "returns": returns,
        "drawdown": dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Candlestick Patterns (pure-python, no TA-Lib)
# ─────────────────────────────────────────────────────────────────────────────
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect 7 common candlestick patterns. Returns df with boolean columns."""
    o = df["Open"]; c = df["Close"]; h = df["High"]; l = df["Low"]
    body = (c - o).abs()
    rng  = h - l

    pat = pd.DataFrame(index=df.index)
    pat["Doji"]             = (body <= rng * 0.1) & (rng > 0)
    pat["Bullish_Engulfing"] = (c.shift(1) < o.shift(1)) & (o < c.shift(1)) & (c > o.shift(1)) & (c > o)
    pat["Bearish_Engulfing"] = (c.shift(1) > o.shift(1)) & (o > c.shift(1)) & (c < o.shift(1)) & (c < o)
    pat["Morning_Star"] = (
        (c.shift(2) < o.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.3) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    )
    pat["Evening_Star"] = (
        (c.shift(2) > o.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.3) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    )
    pat["Piercing"] = (
        (c.shift(1) < o.shift(1)) &
        (o < l.shift(1)) &
        (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1))
    )
    pat["Dark_Cloud"] = (
        (c.shift(1) > o.shift(1)) &
        (o > h.shift(1)) &
        (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1))
    )
    return pat


def get_pattern_summary(pat_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Count and find most recent instance of each pattern."""
    summary = {}
    for col in pat_df.columns:
        hits = pat_df[col][pat_df[col]].index
        summary[col] = {
            "count": int(pat_df[col].sum()),
            "last_date": str(hits[-1].date()) if len(hits) else None,
        }
    # Most recent pattern overall
    last_pattern, last_date = None, None
    for col in pat_df.columns:
        hits = pat_df[col][pat_df[col]].index
        if len(hits):
            if last_date is None or hits[-1] > last_date:
                last_date    = hits[-1]
                last_pattern = col
    summary["_most_recent"] = {"pattern": last_pattern, "date": str(last_date.date()) if last_date else None}
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Seasonality
# ─────────────────────────────────────────────────────────────────────────────
def compute_seasonality(df: pd.DataFrame) -> dict:
    """Monthly and day-of-week return seasonality."""
    rets = df["Close"].pct_change().dropna()
    dates_idx = rets.index

    _dow_map   = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    _month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    dow_rets = {}
    for d, nm in _dow_map.items():
        mask = dates_idx.dayofweek == d
        if mask.sum() > 0:
            s = rets[mask]
            dow_rets[nm] = {"mean": float(s.mean()*100), "positive_pct": float((s>0).mean()*100),
                            "count": int(mask.sum())}

    month_rets = {}
    for m, nm in _month_map.items():
        mask = dates_idx.month == m
        if mask.sum() > 0:
            s = rets[mask]
            month_rets[nm] = {"mean": float(s.mean()*100), "positive_pct": float((s>0).mean()*100),
                              "count": int(mask.sum())}

    best_month  = max(month_rets, key=lambda k: month_rets[k]["mean"]) if month_rets else "N/A"
    worst_month = min(month_rets, key=lambda k: month_rets[k]["mean"]) if month_rets else "N/A"
    best_day    = max(dow_rets, key=lambda k: dow_rets[k]["mean"])   if dow_rets else "N/A"
    worst_day   = min(dow_rets, key=lambda k: dow_rets[k]["mean"])   if dow_rets else "N/A"

    return {
        "monthly": month_rets,
        "daily": dow_rets,
        "best_month": best_month,
        "worst_month": worst_month,
        "best_day": best_day,
        "worst_day": worst_day,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Pillar 2: Technical Score (−1 to +1)
# ─────────────────────────────────────────────────────────────────────────────
def compute_technical_score(df: pd.DataFrame) -> float:
    latest = df.iloc[-1]
    scores = []

    rsi = latest.get("RSI")
    if rsi and not np.isnan(rsi):
        scores.append(-1 if rsi > 70 else (1 if rsi < 30 else (0.5 if rsi < 50 else -0.5)))

    macd = latest.get("MACD"); sig = latest.get("MACD_Sig")
    if macd is not None and sig is not None and not np.isnan(macd):
        scores.append(1 if macd > sig else -1)

    e20 = latest.get("EMA_20"); e50 = latest.get("EMA_50")
    if e20 and e50:
        scores.append(1 if e20 > e50 else -1)

    sma200 = latest.get("SMA_200"); price = latest.get("Close")
    if sma200 and price:
        scores.append(1 if price > sma200 else -1)

    adx = latest.get("ADX")
    if adx and not np.isnan(adx):
        scores.append(0.5 if adx > 25 else 0)

    bb_pct = latest.get("BB_Pct")
    if bb_pct is not None and not np.isnan(bb_pct):
        scores.append(-1 if bb_pct > 0.95 else (1 if bb_pct < 0.05 else 0))

    mfi = latest.get("MFI")
    if mfi and not np.isnan(mfi):
        scores.append(-1 if mfi > 80 else (1 if mfi < 20 else 0))

    willr = latest.get("WILLR")
    if willr and not np.isnan(willr):
        scores.append(1 if willr < -80 else (-1 if willr > -20 else 0))

    return float(np.mean(scores)) if scores else 0.0


def get_risk_flags(df: pd.DataFrame) -> list[str]:
    """Return list of active risk flag strings."""
    flags = []
    latest = df.iloc[-1]

    macd = latest.get("MACD"); msig = latest.get("MACD_Sig")
    if macd is not None and msig is not None and macd < msig:
        flags.append(f"⚠️  MACD Bearish Cross: MACD {macd:.2f} < Signal {msig:.2f}")

    rsi = latest.get("RSI")
    if rsi:
        if rsi > 70: flags.append(f"⚠️  RSI Overbought: {rsi:.1f} > 70")
        if rsi < 30: flags.append(f"⚠️  RSI Oversold: {rsi:.1f} < 30")

    price = latest.get("Close"); sma200 = latest.get("SMA_200")
    if price and sma200 and price < sma200:
        flags.append(f"⚠️  Price below SMA200: {price:.2f} < {sma200:.2f}")

    bb_pct = latest.get("BB_Pct")
    if bb_pct and bb_pct > 0.95:
        flags.append(f"⚠️  Near Bollinger Upper Band (BB%={bb_pct:.2f})")

    return flags
