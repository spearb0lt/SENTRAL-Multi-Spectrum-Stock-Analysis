"""20-strategy backtesting framework + Kelly criterion."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RF_DAILY = 0.06 / 252   # 6% annualized risk-free rate


def _sharpe(rets):
    exc = rets - RF_DAILY
    return (exc.mean() / exc.std()) * 252**0.5 if exc.std() > 0 else 0.0


def _max_dd(cumul):
    return float(((cumul - cumul.cummax()) / cumul.cummax()).min())


def _cagr(cumul, n_days):
    return float(cumul.iloc[-1] ** (252 / n_days) - 1) if n_days > 0 else 0.0


def _win_rate(rets):
    r = rets.dropna(); r = r[r != 0]
    return float(len(r[r > 0]) / len(r)) if len(r) > 0 else 0.0


def _run_one(name, df_bt, signal, returns):
    sig   = signal.reindex(df_bt.index).fillna(0)
    strat = (returns * sig.shift(1)).fillna(0)
    cumul = (1 + strat).cumprod()
    return {
        "Strategy":     name,
        "Total Return": _cagr(cumul, len(strat)) if len(strat) > 0 else 0,
        "Sharpe":       _sharpe(strat.dropna()),
        "Max Drawdown": _max_dd(cumul),
        "Win Rate":     _win_rate(strat),
        "Time in Mkt%": float(sig.mean()),
        "_cumul": cumul,
        "_rets":  strat,
    }


def run_backtest(df: pd.DataFrame) -> dict:
    """Run 20 strategies + Buy & Hold. Returns summary dict."""
    needed = ["Close", "RSI", "MACD", "MACD_Sig", "EMA_20", "EMA_50",
              "SMA_200", "SMA_50", "ADX", "ADX_Pos", "ADX_Neg",
              "Stoch_K", "Stoch_D", "WILLR", "ROC", "MFI",
              "BB_Upper", "BB_Mid", "BB_Lower", "BB_Pct",
              "ATR", "KC_Upper", "KC_Lower", "OBV", "VWAP", "CMF"]
    bt = df[[c for c in needed if c in df.columns]].dropna().copy()
    c    = bt["Close"]
    rets = c.pct_change()

    obv_sma20  = bt["OBV"].rolling(20).mean()
    don20_high = c.shift(1).rolling(20).max()

    s = {}
    s["EMA+MACD+RSI Composite"]   = (((bt.get("EMA_20", c)>bt.get("EMA_50", c)).astype(int)
                                      +(bt["MACD"]>bt["MACD_Sig"]).astype(int)
                                      +((bt["RSI"]>30)&(bt["RSI"]<70)).astype(int))>=2).astype(int)
    s["Golden/Death Cross"]        = (bt.get("SMA_50", c) > bt.get("SMA_200", c)).astype(int)
    s["MACD Crossover"]            = (bt["MACD"] > bt["MACD_Sig"]).astype(int)
    s["RSI Momentum (>50)"]        = (bt["RSI"] > 50).astype(int)
    s["RSI Safe Zone (35-65)"]     = ((bt["RSI"]>35)&(bt["RSI"]<65)).astype(int)
    s["BB Mean Reversion"]         = (bt["BB_Pct"] < 0.30).astype(int)
    s["BB Breakout Momentum"]      = (bt["BB_Pct"] > 0.75).astype(int)
    s["ADX Trend Following"]       = ((bt["ADX"]>25)&(bt.get("ADX_Pos",bt["ADX"])>bt.get("ADX_Neg",bt["ADX"]))).astype(int)
    s["Stochastic Bullish"]        = ((bt["Stoch_K"]>bt["Stoch_D"])&(bt["Stoch_K"]<80)).astype(int)
    s["Williams %R Neutral"]       = ((bt["WILLR"]>-80)&(bt["WILLR"]<-20)).astype(int)
    s["CMF Positive Flow"]         = (bt["CMF"] > 0).astype(int)
    s["OBV Momentum"]              = (bt["OBV"] > obv_sma20).astype(int)
    s["VWAP Momentum"]             = (bt["Close"] > bt["VWAP"]).astype(int)
    s["MFI Healthy Zone"]          = ((bt["MFI"]>25)&(bt["MFI"]<80)).astype(int)
    s["ROC Positive Momentum"]     = (bt["ROC"] > 0).astype(int)
    s["Triple EMA Alignment"]      = ((bt.get("EMA_20",c)>bt.get("EMA_50",c))&(bt.get("EMA_50",c)>bt.get("SMA_200",c))).astype(int)
    s["Donchian 20-Day Breakout"]  = (c > don20_high).astype(int)
    s["Keltner Breakout"]          = (bt["Close"] > bt["KC_Upper"]).astype(int)
    multi = sum((
        (bt.get("EMA_20",c)>bt.get("EMA_50",c)).astype(int),
        (bt["MACD"]>bt["MACD_Sig"]).astype(int),
        (bt["RSI"]>50).astype(int),
        (bt["ADX"]>20).astype(int),
        (bt["CMF"]>0).astype(int),
        (bt["OBV"]>obv_sma20).astype(int),
        (bt["Stoch_K"]>bt["Stoch_D"]).astype(int),
        (bt["Close"]>bt["VWAP"]).astype(int),
    ))
    s["Multi-Confluence (5/8)"]    = (multi >= 5).astype(int)

    results = []
    cumuls  = {}
    strat_rets_map = {}
    for name, sig in s.items():
        r = _run_one(name, bt, sig, rets)
        cumuls[name]         = r.pop("_cumul")
        strat_rets_map[name] = r.pop("_rets")
        results.append(r)

    # Buy & Hold
    bh_cumul = (1 + rets.fillna(0)).cumprod()
    results.append({
        "Strategy": "★ Buy & Hold",
        "Total Return": _cagr(bh_cumul, len(bt)),
        "Sharpe": _sharpe(rets.dropna()),
        "Max Drawdown": _max_dd(bh_cumul),
        "Win Rate": _win_rate(rets),
        "Time in Mkt%": 1.0,
    })
    cumuls["★ Buy & Hold"] = bh_cumul

    df_results = pd.DataFrame(results).set_index("Strategy").sort_values("Sharpe", ascending=False)

    best_name = next(n for n in df_results.index if n != "★ Buy & Hold")
    best_rets = strat_rets_map.get(best_name, rets.fillna(0))

    return {
        "df_results": df_results,
        "cumuls": cumuls,
        "best_name": best_name,
        "best_rets": best_rets,
        "bh_cumul": bh_cumul,
        "bt_index": bt.index,
    }


def compute_kelly(best_rets: pd.Series, current_price: float,
                   atr: float, portfolio_size: int = 100_000) -> dict:
    """Kelly criterion position sizing from best strategy returns."""
    r = best_rets.dropna()
    r = r[r != 0]

    wins    = r[r > 0]
    losses  = r[r < 0]
    p_win   = float(len(wins) / len(r)) if len(r) else 0.5
    avg_win  = float(wins.mean()) if len(wins) else 0
    avg_loss = abs(float(losses.mean())) if len(losses) else 1e-6
    b        = avg_win / avg_loss
    kelly_f  = (p_win * b - (1 - p_win)) / b if b > 0 else 0
    half_k   = max(0, kelly_f / 2)

    stop_1x = current_price - 1.5 * atr
    stop_2x = current_price - 2.0 * atr
    tgt_1r  = current_price + 1.5 * atr
    tgt_2r  = current_price + 2.0 * atr

    alloc   = portfolio_size * half_k
    n_shares = int(alloc // current_price) if current_price > 0 else 0

    return {
        "p_win": p_win, "avg_win": avg_win, "avg_loss": avg_loss,
        "b_ratio": b, "kelly_f": kelly_f, "half_kelly": half_k,
        "stop_1x": stop_1x, "stop_2x": stop_2x,
        "target_1r": tgt_1r, "target_2r": tgt_2r,
        "allocation": alloc, "n_shares": n_shares,
        "ev_per_trade": p_win * avg_win - (1-p_win) * avg_loss,
    }
