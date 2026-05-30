"""Peer comparison: LLM-powered discovery → hard-coded fallback."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ── Hard-coded fallback peers by sector ────────────────────────────────────────
SECTOR_PEERS = {
    "Industrials":            ["HAL.NS","BEL.NS","BEML.NS","MTAR.NS","BHEL.NS","BDL.NS","COCHINSHIP.NS","GRSE.NS"],
    "Technology":             ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS"],
    "Financials":             ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BAJFINANCE.NS","INDUSINDBK.NS"],
    "Consumer Defensive":     ["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS"],
    "Healthcare":             ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS","LUPIN.NS","TORNTPHARM.NS"],
    "Energy":                 ["RELIANCE.NS","ONGC.NS","IOC.NS","BPCL.NS","GAIL.NS","HINDPETRO.NS","OIL.NS"],
    "Basic Materials":        ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","COALINDIA.NS","NMDC.NS","SAIL.NS"],
    "Utilities":              ["NTPC.NS","POWERGRID.NS","ADANIGREEN.NS","TATAPOWER.NS","SJVN.NS","CESC.NS","TORNTPOWER.NS"],
    "Communication Services": ["BHARTIARTL.NS","IDEA.NS","INDUSTOWER.NS","TATACOMM.NS"],
    "Consumer Cyclical":      ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","M&M.NS","EICHERMOT.NS","TVS-E.NS"],
    "Real Estate":            ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","OBEROIRLTY.NS","SOBHA.NS","PHOENIXLTD.NS"],
    # US markets
    "Technology (US)":        ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSM"],
    "Financials (US)":        ["JPM","BAC","WFC","GS","MS","C","BLK"],
}

METRIC_COLS = ["P/E", "P/B", "ROE", "Div Yield", "Mkt Cap (Cr)", "Beta", "EV/EBITDA"]


# ── LLM-powered peer discovery ─────────────────────────────────────────────────
def _llm_discover_peers(
    ticker: str,
    company_name: str,
    sector: str,
    exchange: str,
    groq_key: str = "",
    gemini_key: str = "",
) -> list[str]:
    """Ask LLM to identify 8 peer Yahoo Finance ticker symbols.
    Returns empty list on failure so caller can use hard-coded fallback."""
    prompt = (
        f"List exactly 8 publicly traded peer/competitor companies for {company_name} "
        f"({ticker}), a {sector} company listed on {exchange}. "
        f"Return ONLY a comma-separated list of Yahoo Finance ticker symbols "
        f"(e.g. AAPL,MSFT,GOOGL). Include {ticker} first. "
        f"No explanations, no numbering, no extra text — symbols only."
    )

    # 1. Try Groq (fast, free tier)
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.1,
                    )
                    raw = resp.choices[0].message.content.strip()
                    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
                    # Basic sanity: at least 3 symbols, no spaces inside a token
                    tickers = [t for t in tickers if " " not in t and len(t) <= 15]
                    if len(tickers) >= 3:
                        return tickers[:9]
                except Exception:
                    continue
        except ImportError:
            pass

    # 2. Try Gemini fallback
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp  = model.generate_content(prompt)
                    raw   = resp.text.strip()
                    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
                    tickers = [t for t in tickers if " " not in t and len(t) <= 15]
                    if len(tickers) >= 3:
                        return tickers[:9]
                except Exception:
                    continue
        except ImportError:
            pass

    return []  # both failed → caller uses hard-coded fallback


# ── Hard-coded fallback resolver ───────────────────────────────────────────────
def _hardcoded_peers(ticker: str, sector: str) -> list[str]:
    """Return hard-coded sector peers (includes ticker as first item)."""
    auto = [ticker]
    for k, v in SECTOR_PEERS.items():
        if ticker in v or k.lower() in sector.lower():
            for sym in v:
                if sym != ticker:
                    auto.append(sym)
            break
    return auto


# ── Main cached function ───────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_peer_data(
    ticker: str,
    sector: str,
    company_name: str = "",
    exchange: str = "NSE",
    groq_key: str = "",
    gemini_key: str = "",
) -> tuple[list[str], pd.DataFrame, str]:
    """Discover peers (LLM-first, hard-coded fallback) and return comparison df.

    Returns: (peer_list, df_peers, discovery_method)
    """
    # --- Step 1: Try LLM discovery ---
    llm_peers = _llm_discover_peers(
        ticker, company_name or ticker, sector, exchange, groq_key, gemini_key
    )
    if llm_peers and len(llm_peers) >= 3:
        peer_list = llm_peers
        method = "LLM"
    else:
        # --- Step 2: Hard-coded sector fallback ---
        peer_list = _hardcoded_peers(ticker, sector)
        method = "sector-database"

    # Ensure target ticker is first
    if ticker in peer_list:
        peer_list.remove(ticker)
    peer_list = [ticker] + peer_list[:8]

    # --- Step 3: Fetch metrics for all peers ---
    rows = []
    for sym in peer_list:
        try:
            info = yf.Ticker(sym).info
            mktcap = info.get("marketCap")
            ebitda = info.get("ebitda")
            ev     = info.get("enterpriseValue")
            row = {
                "Ticker":       sym,
                "Name":         info.get("shortName", sym)[:25],
                "Price":        info.get("currentPrice") or info.get("regularMarketPrice"),
                "P/E":          round(info.get("trailingPE", 0), 1) if info.get("trailingPE") else None,
                "P/B":          round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else None,
                "ROE (%)":      round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
                "ROCE (%)":     round(info.get("returnOnAssets",  0) * 100, 1) if info.get("returnOnAssets")  else None,
                "Div Yield (%)":round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
                "Mkt Cap (Cr)": round(mktcap / 1e7, 0) if mktcap else None,
                "EV/EBITDA":    round(ev / ebitda, 1)   if ev and ebitda and ebitda > 0 else None,
                "Beta":         round(info.get("beta", 0), 2) if info.get("beta") else None,
                "52W High":     info.get("fiftyTwoWeekHigh"),
                "52W Low":      info.get("fiftyTwoWeekLow"),
            }
            rows.append(row)
        except Exception:
            pass

    df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()
    return peer_list, df, method
