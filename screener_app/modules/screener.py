"""Multi-stock screener engine."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ── Default universe ──────────────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    # Industrials / Defence
    "HAL.NS","BEL.NS","BEML.NS","BHEL.NS","BDL.NS","MTAR.NS","COCHINSHIP.NS","GRSE.NS",
    # IT
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS",
    # Financials
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BAJFINANCE.NS","INDUSINDBK.NS",
    # FMCG
    "HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS",
    # Pharma
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS","LUPIN.NS",
    # Energy
    "RELIANCE.NS","ONGC.NS","IOC.NS","BPCL.NS","GAIL.NS",
    # Metals
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","COALINDIA.NS",
    # Utilities
    "NTPC.NS","POWERGRID.NS","ADANIGREEN.NS","TATAPOWER.NS","SJVN.NS",
    # Auto
    "MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","M&M.NS","EICHERMOT.NS",
    # Real Estate
    "DLF.NS","GODREJPROP.NS","PRESTIGE.NS","OBEROIRLTY.NS",
    # Large caps
    "ADANIENT.NS","ADANIPORTS.NS","ASIANPAINT.NS","ULTRACEMCO.NS","TITAN.NS","ITC.NS",
]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_data(tickers: tuple) -> pd.DataFrame:
    """Fetch key metrics for all tickers in universe. Returns DataFrame."""
    rows = []
    for sym in tickers:
        try:
            i   = yf.Ticker(sym).info
            mc_ = (i.get("marketCap") or 0) / 1e7
            roe_= (i.get("returnOnEquity") or 0) * 100
            nm_ = (i.get("profitMargins") or 0) * 100
            rg_ = (i.get("revenueGrowth") or 0) * 100
            dy_ = (i.get("dividendYield") or 0) * 100
            ev_ = i.get("enterpriseValue")
            eb_ = i.get("ebitda")
            rows.append({
                "Ticker":        sym,
                "Name":          (i.get("shortName") or sym)[:25],
                "Sector":        i.get("sector", "—"),
                "Price":         i.get("currentPrice") or i.get("regularMarketPrice"),
                "Mkt Cap (Cr)":  round(mc_, 0),
                "P/E":           round(i.get("trailingPE", 0), 1) if i.get("trailingPE") else None,
                "P/B":           round(i.get("priceToBook", 0), 2) if i.get("priceToBook") else None,
                "ROE %":         round(roe_, 1),
                "Net Margin %":  round(nm_, 1),
                "Rev Growth %":  round(rg_, 1),
                "EV/EBITDA":     round(ev_/eb_, 1) if ev_ and eb_ and eb_ > 0 else None,
                "Div Yield %":   round(dy_, 2),
                "D/E":           round(i.get("debtToEquity", 0), 2) if i.get("debtToEquity") else None,
                "Current Ratio": round(i.get("currentRatio", 0), 2) if i.get("currentRatio") else None,
                "Beta":          round(i.get("beta", 0), 2) if i.get("beta") else None,
                "52W High":      i.get("fiftyTwoWeekHigh"),
                "52W Low":       i.get("fiftyTwoWeekLow"),
            })
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply screener filters to a DataFrame. Returns filtered copy."""
    if df.empty:
        return df
    out = df.copy()
    for col, (op, val) in filters.items():
        if col not in out.columns or val is None:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if op == "<=":
            out = out[out[col].isna() | (out[col] <= val)]
        elif op == ">=":
            out = out[out[col].isna() | (out[col] >= val)]
        elif op == "==":
            out = out[out[col] == val]
    return out
