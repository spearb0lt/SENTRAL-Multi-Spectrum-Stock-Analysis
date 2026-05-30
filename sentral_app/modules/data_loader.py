"""Data download and feature engineering module."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import streamlit as st
# Note: stockstats not used here — removed to keep deps lean


@st.cache_data(ttl=86400, show_spinner=False)
def download_stock_data(ticker: str, period: str):
    """Download OHLCV + financials from yfinance. Cached 24h per ticker."""
    stock = yf.Ticker(ticker)
    info  = stock.info

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    hist = stock.history(period=period)
    fin  = stock.financials.T
    bal  = stock.balance_sheet.T
    cf   = stock.cashflow.T

    company_name = info.get("longName", ticker)
    sector       = info.get("sector", "N/A")
    industry     = info.get("industry", "N/A")
    currency     = info.get("currency", "INR")
    exchange     = info.get("exchange", "N/A")
    currency_sym = "₹" if currency == "INR" else "$"

    # NOTE: yfinance.Ticker object (stock) is NOT pickle-serializable, so it is
    # intentionally excluded from this cached return value.
    return {
        "info": info, "data": data, "hist": hist,
        "fin": fin, "bal": bal, "cf": cf,
        "company_name": company_name, "sector": sector, "industry": industry,
        "currency": currency, "currency_sym": currency_sym, "exchange": exchange,
    }


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add 35 technical indicators to OHLCV dataframe."""
    df = data.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    for w in [10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = c.rolling(w).mean()
    df["EMA_10"]  = c.ewm(span=10,  adjust=False).mean()
    df["EMA_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()

    # Momentum
    rsi_obj     = ta.momentum.RSIIndicator(c, window=14)
    df["RSI"]   = rsi_obj.rsi()
    stoch       = ta.momentum.StochasticOscillator(h, l, c)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["WILLR"]   = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    df["ROC"]     = ta.momentum.ROCIndicator(c).roc()

    # MACD
    macd_obj     = ta.trend.MACD(c)
    df["MACD"]      = macd_obj.macd()
    df["MACD_Sig"]  = macd_obj.macd_signal()
    df["MACD_Hist"] = macd_obj.macd_diff()

    # Volatility
    bb  = ta.volatility.BollingerBands(c)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"]   = bb.bollinger_mavg()
    df["BB_Pct"]   = bb.bollinger_pband()
    df["BB_Width"] = bb.bollinger_wband()
    atr_obj      = ta.volatility.AverageTrueRange(h, l, c)
    df["ATR"]    = atr_obj.average_true_range()
    kc           = ta.volatility.KeltnerChannel(h, l, c)
    df["KC_Upper"] = kc.keltner_channel_hband()
    df["KC_Lower"] = kc.keltner_channel_lband()

    # Volume
    df["OBV"]  = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["CMF"]  = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
    df["MFI"]  = ta.volume.MFIIndicator(h, l, c, v).money_flow_index()
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()

    # Trend
    adx_obj      = ta.trend.ADXIndicator(h, l, c)
    df["ADX"]    = adx_obj.adx()
    df["ADX_Pos"] = adx_obj.adx_pos()
    df["ADX_Neg"] = adx_obj.adx_neg()
    df["CCI"]    = ta.trend.CCIIndicator(h, l, c).cci()
    df["DPO"]    = ta.trend.DPOIndicator(c).dpo()
    df["Ichimoku_A"] = ta.trend.IchimokuIndicator(h, l).ichimoku_a()
    df["Ichimoku_B"] = ta.trend.IchimokuIndicator(h, l).ichimoku_b()

    # Other
    df["Ulcer"] = ta.volatility.UlcerIndex(c).ulcer_index()

    # Returns
    df["Return_1d"]  = c.pct_change(1)
    df["Return_5d"]  = c.pct_change(5)
    df["Return_20d"] = c.pct_change(20)

    df.dropna(how="all", inplace=True)
    return df
