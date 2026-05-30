"""Peer discovery: LLM-first, hard-coded-sector fallback."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

SECTOR_PEERS = {
    "Industrials":            ["HAL.NS","BEL.NS","BEML.NS","MTAR.NS","BHEL.NS","BDL.NS","COCHINSHIP.NS","GRSE.NS"],
    "Technology":             ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS"],
    "Financials":             ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BAJFINANCE.NS"],
    "Consumer Defensive":     ["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","GODREJCP.NS"],
    "Healthcare":             ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS","LUPIN.NS"],
    "Energy":                 ["RELIANCE.NS","ONGC.NS","IOC.NS","BPCL.NS","GAIL.NS","HINDPETRO.NS"],
    "Basic Materials":        ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","COALINDIA.NS","NMDC.NS"],
    "Utilities":              ["NTPC.NS","POWERGRID.NS","ADANIGREEN.NS","TATAPOWER.NS","SJVN.NS","CESC.NS"],
    "Communication Services": ["BHARTIARTL.NS","IDEA.NS","INDUSTOWER.NS","TATACOMM.NS"],
    "Consumer Cyclical":      ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","M&M.NS","EICHERMOT.NS"],
    "Real Estate":            ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","OBEROIRLTY.NS","SOBHA.NS"],
    "Technology (US)":        ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSM"],
    "Financials (US)":        ["JPM","BAC","WFC","GS","MS","C","BLK"],
}

GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]


def _llm_discover(ticker, company, sector, exchange, groq_key, gemini_key):
    prompt = (
        f"List exactly 8 publicly traded peer companies for {company} ({ticker}), "
        f"a {sector} company on {exchange}. Return ONLY comma-separated Yahoo Finance "
        f"ticker symbols. Include {ticker} first. No explanations, symbols only."
    )
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            for model in GROQ_MODELS:
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}],
                        max_tokens=200, temperature=0.1)
                    raw = resp.choices[0].message.content.strip()
                    tks = [t.strip().upper() for t in raw.split(",")
                           if t.strip() and " " not in t.strip() and len(t.strip()) <= 15]
                    if len(tks) >= 3:
                        return tks[:9], f"Groq ({model})"
                except Exception:
                    continue
        except ImportError:
            pass

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            for model_n in GEMINI_MODELS:
                try:
                    m    = genai.GenerativeModel(model_n)
                    resp = m.generate_content(prompt)
                    raw  = resp.text.strip()
                    tks  = [t.strip().upper() for t in raw.split(",")
                            if t.strip() and " " not in t.strip() and len(t.strip()) <= 15]
                    if len(tks) >= 3:
                        return tks[:9], f"Gemini ({model_n})"
                except Exception:
                    continue
        except ImportError:
            pass

    return [], "hard-coded"


@st.cache_data(ttl=86400, show_spinner=False)
def get_peer_data(
    ticker: str,
    sector: str,
    company_name: str = "",
    exchange: str = "NSE",
    groq_key: str = "",
    gemini_key: str = "",
) -> tuple[list, pd.DataFrame, str]:
    llm_peers, method = _llm_discover(ticker, company_name or ticker, sector, exchange, groq_key, gemini_key)

    if llm_peers and len(llm_peers) >= 3:
        peer_list = llm_peers
    else:
        peer_list = [ticker]
        for k, v in SECTOR_PEERS.items():
            if ticker in v or k.lower() in sector.lower():
                peer_list += [s for s in v if s != ticker][:8]
                break
        method = "sector-database"

    if ticker in peer_list:
        peer_list.remove(ticker)
    peer_list = [ticker] + peer_list[:8]

    rows = []
    for sym in peer_list:
        try:
            pi  = yf.Ticker(sym).info
            mc_ = pi.get("marketCap")
            ev_ = pi.get("enterpriseValue")
            eb_ = pi.get("ebitda")
            rows.append({
                "Ticker":        sym,
                "Name":          (pi.get("shortName") or sym)[:22],
                "Price":         pi.get("currentPrice") or pi.get("regularMarketPrice"),
                "Mkt Cap (Cr)":  round(mc_ / 1e7, 0) if mc_ else None,
                "P/E":           round(pi.get("trailingPE", 0), 1) if pi.get("trailingPE") else None,
                "P/B":           round(pi.get("priceToBook", 0), 2) if pi.get("priceToBook") else None,
                "ROE %":         round((pi.get("returnOnEquity") or 0) * 100, 1),
                "Net Margin %":  round((pi.get("profitMargins") or 0) * 100, 1),
                "EV/EBITDA":     round(ev_ / eb_, 1) if ev_ and eb_ and eb_ > 0 else None,
                "Div Yield %":   round((pi.get("dividendYield") or 0) * 100, 2),
                "D/E":           round(pi.get("debtToEquity", 0), 2) if pi.get("debtToEquity") else None,
                "Beta":          round(pi.get("beta", 0), 2) if pi.get("beta") else None,
            })
        except Exception:
            pass

    df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()
    return peer_list, df, method
