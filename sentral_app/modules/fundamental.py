"""Fundamental analysis: metrics, DCF, Altman Z-Score, Piotroski F-Score."""
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Extended Fundamental Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_fundamental_metrics(info: dict, fin: pd.DataFrame,
                                 bal: pd.DataFrame, cf: pd.DataFrame,
                                 currency_sym: str) -> dict:
    """Return dict of 38 fundamental metrics."""
    i = info
    metrics = {}

    # Valuation
    metrics["Market Cap"]         = i.get("marketCap")
    metrics["Enterprise Value"]   = i.get("enterpriseValue")
    metrics["P/E Trailing"]       = i.get("trailingPE")
    metrics["P/E Forward"]        = i.get("forwardPE")
    metrics["P/B Ratio"]          = i.get("priceToBook")
    metrics["P/S Ratio"]          = i.get("priceToSalesTrailing12Months")
    metrics["EV/EBITDA"]          = i.get("enterpriseToEbitda")
    metrics["EV/Revenue"]         = i.get("enterpriseToRevenue")

    # Profitability
    metrics["Gross Margin"]       = i.get("grossMargins")
    metrics["Operating Margin"]   = i.get("operatingMargins")
    metrics["Net Margin"]         = i.get("profitMargins")
    metrics["ROE"]                = i.get("returnOnEquity")
    metrics["ROA"]                = i.get("returnOnAssets")
    metrics["ROIC"]               = _safe_roic(fin, bal)

    # Growth
    metrics["Revenue Growth (YoY)"]    = i.get("revenueGrowth")
    metrics["Earnings Growth (YoY)"]   = i.get("earningsGrowth")

    # Leverage & Liquidity
    metrics["Debt/Equity"]        = i.get("debtToEquity")
    metrics["Current Ratio"]      = i.get("currentRatio")
    metrics["Quick Ratio"]        = i.get("quickRatio")
    metrics["Interest Coverage"]  = _safe_interest_coverage(fin)

    # Per Share
    metrics["EPS (Trailing)"]     = i.get("trailingEps")
    metrics["EPS (Forward)"]      = i.get("forwardEps")
    metrics["Book Value/Share"]   = i.get("bookValue")
    metrics["Dividend Yield"]     = i.get("dividendYield")
    metrics["Payout Ratio"]       = i.get("payoutRatio")

    # Cash Flow
    metrics["FCF Yield"]          = _safe_fcf_yield(cf, i)
    metrics["Operating CF/Share"] = _safe_ocf_share(cf, i)

    # Others
    metrics["Beta"]               = i.get("beta")
    metrics["52W High"]           = i.get("fiftyTwoWeekHigh")
    metrics["52W Low"]            = i.get("fiftyTwoWeekLow")
    metrics["Shares Outstanding"] = i.get("sharesOutstanding")
    metrics["Float Shares"]       = i.get("floatShares")

    return metrics


def _safe_roic(fin, bal):
    try:
        nopat = fin.iloc[0].get("Operating Income", np.nan)
        ic    = bal.iloc[0].get("Total Assets", np.nan) - bal.iloc[0].get("Current Liabilities", np.nan)
        return float(nopat) / float(ic) if ic and ic != 0 else np.nan
    except Exception:
        return np.nan


def _safe_interest_coverage(fin):
    try:
        ebit     = fin.iloc[0].get("Operating Income", np.nan)
        interest = fin.iloc[0].get("Interest Expense", np.nan)
        if interest and interest != 0:
            return float(ebit) / abs(float(interest))
    except Exception:
        pass
    return np.nan


def _safe_fcf_yield(cf, info):
    try:
        mktcap = info.get("marketCap")
        fcf    = _get_fcf(cf)
        if mktcap and fcf:
            return fcf / mktcap
    except Exception:
        pass
    return np.nan


def _safe_ocf_share(cf, info):
    try:
        shares = info.get("sharesOutstanding")
        if isinstance(cf.index[0], pd.Timestamp):
            ocf = cf.iloc[0].get("Operating Cash Flow", np.nan)
        else:
            ocf = cf.get("Operating Cash Flow", pd.Series([np.nan])).iloc[0]
        if shares and shares > 0:
            return float(ocf) / float(shares)
    except Exception:
        pass
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  DCF Valuation
# ─────────────────────────────────────────────────────────────────────────────
def _get_fcf(cf: pd.DataFrame) -> float | None:
    """Extract the most recent FCF from either orientation of cf DataFrame."""
    try:
        if isinstance(cf.index[0], pd.Timestamp):
            # Transposed: index=dates, columns=metric names
            for col in cf.columns:
                if "free cash flow" in str(col).lower():
                    val = cf[col].dropna().iloc[0]
                    return float(val) if not np.isnan(val) else None
            # Fallback: OCF - CapEx
            ocf  = cf.get("Operating Cash Flow",  pd.Series(dtype=float)).dropna()
            capx = cf.get("Capital Expenditure",  pd.Series(dtype=float)).dropna()
            if len(ocf) and len(capx):
                return float(ocf.iloc[0]) - abs(float(capx.iloc[0]))
        else:
            for idx in cf.index:
                if "free cash flow" in str(idx).lower():
                    val = cf.loc[idx].dropna().iloc[0]
                    return float(val) if not np.isnan(val) else None
    except Exception:
        pass
    return None


def run_dcf(cf: pd.DataFrame, info: dict, currency_sym: str,
            wacc: float = 0.10, stage1_g: float = 0.12,
            stage2_g: float = 0.07, terminal_g: float = 0.04,
            n_stage1: int = 5) -> dict:
    """Run two-stage DCF. Returns dict with result keys."""
    result = {"fcf": None, "intrinsic": None, "upside_pct": None,
              "signal": "N/A", "error": None}
    try:
        fcf_raw = _get_fcf(cf)
        if fcf_raw is None:
            result["error"] = "FCF unavailable in financials"
            return result

        fcf_cr = fcf_raw / 1e7   # paise → crores (for INR); for USD just /1e6
        if abs(fcf_raw) > 1e10:   # already in base currency units
            fcf_cr = fcf_raw / 1e7

        shares = info.get("sharesOutstanding", 0)
        price  = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        net_debt_val = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)

        # Stage 1 — high growth
        pv_fcfs, fcf_t = [], fcf_cr
        for _ in range(n_stage1):
            fcf_t *= (1 + stage1_g)
            pv_fcfs.append(fcf_t / (1 + wacc) ** (_ + 1))

        # Stage 2 — moderate growth (5 more years)
        for j in range(5):
            fcf_t *= (1 + stage2_g)
            pv_fcfs.append(fcf_t / (1 + wacc) ** (n_stage1 + j + 1))

        # Terminal value
        tv    = fcf_t * (1 + terminal_g) / (wacc - terminal_g)
        pv_tv = tv / (1 + wacc) ** (n_stage1 + 5)

        equity_val   = sum(pv_fcfs) + pv_tv - (net_debt_val / 1e7)
        n_shares_cr  = shares / 1e7
        intrinsic    = (equity_val / n_shares_cr) * 1e7 / shares if n_shares_cr > 0 else 0
        upside       = (intrinsic / price - 1) * 100 if price else 0

        result.update({
            "fcf": fcf_cr,
            "intrinsic": intrinsic,
            "current_price": price,
            "upside_pct": upside,
            "signal": "UNDERVALUED" if upside > 0 else "OVERVALUED",
        })
    except Exception as e:
        result["error"] = str(e)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Altman Z-Score
# ─────────────────────────────────────────────────────────────────────────────
def run_altman_z(bal: pd.DataFrame, fin: pd.DataFrame, info: dict) -> dict:
    """Returns Altman Z-Score and zone classification."""
    result = {"z_score": np.nan, "zone": "N/A", "breakdown": {}}
    try:
        b = bal.iloc[0]; f = fin.iloc[0]

        def g(df, *keys, default=0):
            for k in keys:
                for col in df.index if hasattr(df, 'index') else []:
                    if k.lower() in str(col).lower():
                        try: return float(df[col])
                        except Exception: pass
            return default

        ta_val  = g(b, "Total Assets", "totalAssets") or 1
        tl      = g(b, "Total Liabilities", "totalLiab")
        ca      = g(b, "Current Assets", "currentAssets")
        cl      = g(b, "Current Liabilities", "currentLiab")
        re      = g(b, "Retained Earnings", "retainedEarnings")
        ebit    = g(f, "Operating Income", "EBIT", "Ebit")
        equity  = g(b, "Common Stock Equity", "Stockholders Equity", "Total Equity")
        sales   = g(f, "Total Revenue", "Revenue")
        mktcap  = info.get("marketCap", 0) or 0

        x1 = (ca - cl) / ta_val
        x2 = re / ta_val
        x3 = ebit / ta_val
        x4 = mktcap / tl if tl else 0
        x5 = sales / ta_val

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        zone = "SAFE" if z > 2.99 else ("GREY" if z > 1.81 else "DISTRESS")

        result = {"z_score": round(z, 3), "zone": zone,
                  "breakdown": {"X1(WC/TA)": round(x1,3), "X2(RE/TA)": round(x2,3),
                                 "X3(EBIT/TA)": round(x3,3), "X4(MktCap/TL)": round(x4,3),
                                 "X5(Sales/TA)": round(x5,3)}}
    except Exception as e:
        result["error"] = str(e)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Piotroski F-Score
# ─────────────────────────────────────────────────────────────────────────────
def run_piotroski(bal: pd.DataFrame, fin: pd.DataFrame, cf: pd.DataFrame) -> dict:
    """Returns Piotroski F-Score (0-9) and signal."""
    criteria = {}
    try:
        if len(bal) >= 2 and len(fin) >= 2:
            b0, b1 = bal.iloc[0], bal.iloc[1]
            f0, f1 = fin.iloc[0], fin.iloc[1]

            def _get(s, *keys):
                for k in keys:
                    for col in s.index:
                        if k.lower() in str(col).lower():
                            try: return float(s[col])
                            except Exception: pass
                return 0.0

            ta0  = _get(b0,"Total Assets") or 1
            ta1  = _get(b1,"Total Assets") or 1
            roa0 = _get(f0,"Net Income") / ta0
            roa1 = _get(f1,"Net Income") / ta1

            # Get OCF
            if isinstance(cf.index[0], pd.Timestamp):
                ocf = cf.get("Operating Cash Flow", pd.Series(dtype=float)).dropna()
                ocf0 = float(ocf.iloc[0]) if len(ocf) > 0 else 0
            else:
                ocf_row = [c for c in cf.index if "operating" in str(c).lower() and "cash" in str(c).lower()]
                ocf0 = float(cf.loc[ocf_row[0]].iloc[0]) if ocf_row else 0

            criteria["F1_ROA_positive"]    = 1 if roa0 > 0 else 0
            criteria["F2_OCF_positive"]    = 1 if ocf0 > 0 else 0
            criteria["F3_ROA_improving"]   = 1 if roa0 > roa1 else 0
            criteria["F4_Accruals_low"]    = 1 if ocf0 / ta0 > roa0 else 0

            lev0 = _get(b0,"Total Debt","Long Term Debt") / ta0
            lev1 = _get(b1,"Total Debt","Long Term Debt") / ta1
            criteria["F5_Leverage_lower"]  = 1 if lev0 < lev1 else 0

            cr0  = _get(b0,"Current Assets") / (_get(b0,"Current Liabilities") or 1)
            cr1  = _get(b1,"Current Assets") / (_get(b1,"Current Liabilities") or 1)
            criteria["F6_Liquidity_up"]    = 1 if cr0 > cr1 else 0
            criteria["F7_No_dilution"]     = 1  # assume no new shares

            gm0 = _get(f0,"Gross Profit") / (_get(f0,"Total Revenue") or 1)
            gm1 = _get(f1,"Gross Profit") / (_get(f1,"Total Revenue") or 1)
            criteria["F8_Gross_margin_up"] = 1 if gm0 > gm1 else 0

            at0 = _get(f0,"Total Revenue") / ta0
            at1 = _get(f1,"Total Revenue") / ta1
            criteria["F9_Asset_turnover_up"] = 1 if at0 > at1 else 0

    except Exception:
        pass

    score  = sum(criteria.values())
    signal = "STRONG" if score >= 7 else ("MODERATE" if score >= 4 else "WEAK")
    return {"score": score, "signal": signal, "criteria": criteria}


# ─────────────────────────────────────────────────────────────────────────────
#  Pillar 1: Fundamental Score (−1 to +1)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fundamental_score(info: dict, piotroski_score: int, z_score: float) -> float:
    scores = []
    pe = info.get("trailingPE")
    if pe:
        scores.append(0.5 if pe < 15 else (-0.5 if pe > 40 else 0))
    pb = info.get("priceToBook")
    if pb:
        scores.append(0.5 if pb < 1.5 else (-0.5 if pb > 5 else 0))
    roe = info.get("returnOnEquity", 0)
    if roe:
        scores.append(1 if roe > 0.20 else (0.5 if roe > 0.10 else (-0.5 if roe < 0 else 0)))
    npm = info.get("profitMargins", 0)
    if npm:
        scores.append(1 if npm > 0.20 else (0.5 if npm > 0.05 else (-0.5 if npm < 0 else 0)))
    scores.append((piotroski_score / 9) * 2 - 1)
    if not np.isnan(z_score):
        scores.append(1 if z_score > 2.99 else (-1 if z_score < 1.81 else 0))
    de = info.get("debtToEquity")
    if de is not None:
        scores.append(0.5 if de < 0.5 else (-0.5 if de > 2 else 0))
    rg = info.get("revenueGrowth", 0)
    if rg:
        scores.append(1 if rg > 0.20 else (0.5 if rg > 0.05 else (-0.5 if rg < 0 else 0)))
    return float(np.mean(scores)) if scores else 0.0
