"""Data loading and feature computation for Screener app."""
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import streamlit as st
from datetime import datetime


@st.cache_data(ttl=3600, show_spinner=False)
def download_full_data(ticker: str, period: str = "5y") -> dict:
    """Download all yfinance data for a ticker. Returns dict with all DataFrames."""
    stock = yf.Ticker(ticker)
    info  = stock.info

    hist = stock.history(period=period, auto_adjust=True)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    def safe_df(fn):
        try:
            df = fn()
            if df is not None and not df.empty:
                df = df.T
                df.index = pd.to_datetime(df.index)
                return df.sort_index()
        except Exception:
            pass
        return pd.DataFrame()

    fin  = safe_df(lambda: stock.financials)
    bal  = safe_df(lambda: stock.balance_sheet)
    cf   = safe_df(lambda: stock.cashflow)
    qfin = safe_df(lambda: stock.quarterly_financials)
    qbal = safe_df(lambda: stock.quarterly_balance_sheet)
    qcf  = safe_df(lambda: stock.quarterly_cashflow)

    try:
        actions = stock.actions
        if actions is not None:
            actions.index = pd.to_datetime(actions.index).tz_localize(None)
    except Exception:
        actions = pd.DataFrame()

    try:
        holders_inst  = stock.institutional_holders
        holders_major = stock.major_holders
    except Exception:
        holders_inst = holders_major = pd.DataFrame()

    try:
        calendar = stock.calendar
    except Exception:
        calendar = None

    return {
        "info":          info,
        "hist":          hist,
        "fin":           fin,
        "bal":           bal,
        "cf":            cf,
        "qfin":          qfin,
        "qbal":          qbal,
        "qcf":           qcf,
        "actions":       actions,
        "holders_inst":  holders_inst,
        "holders_major": holders_major,
        "calendar":      calendar,
        "company_name":  info.get("longName") or info.get("shortName") or ticker,
        "sector":        info.get("sector", "Unknown"),
        "industry":      info.get("industry", "Unknown"),
        "exchange":      info.get("exchange", "N/A"),
        "currency":      info.get("currency", "INR"),
        "currency_sym":  "₹" if info.get("currency","INR") == "INR" else "$",
    }


@st.cache_data(ttl=86400, show_spinner=False)
def download_benchmark(bench: str, period: str = "5y") -> pd.DataFrame:
    try:
        bh = yf.Ticker(bench).history(period=period, auto_adjust=True)
        bh.index = pd.to_datetime(bh.index).tz_localize(None)
        return bh
    except Exception:
        return pd.DataFrame()


def compute_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to OHLCV history."""
    df = hist.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    for w in [10, 20, 50, 100, 200]:
        df[f"SMA{w}"] = c.rolling(w).mean()
    df["EMA20"]  = c.ewm(span=20, adjust=False).mean()
    df["EMA50"]  = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()

    # Momentum
    df["RSI"]   = ta.momentum.RSIIndicator(close=c, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["ROC"]   = ta.momentum.ROCIndicator(close=c, window=10).roc()
    df["Williams_R"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()

    # Trend
    macd = ta.trend.MACD(close=c)
    df["MACD"]      = macd.macd()
    df["MACD_Sig"]  = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(h, l, c)
    df["ADX"]     = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # Volatility
    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Pct"]    = bb.bollinger_pband()
    df["ATR"]       = ta.volatility.AverageTrueRange(h, l, c).average_true_range()

    # Volume
    df["OBV"]  = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["CMF"]  = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()
    df["MFI"]  = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()
    try:
        df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()
    except Exception:
        pass

    # Ichimoku
    try:
        ich = ta.trend.IchimokuIndicator(h, l)
        df["Ichi_A"] = ich.ichimoku_a()
        df["Ichi_B"] = ich.ichimoku_b()
    except Exception:
        pass

    return df


def _safe_col(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(dtype=float, index=df.index)
