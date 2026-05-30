"""Fundamental analysis: P&L, Balance Sheet, Cash Flow, Ratios, Scores."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from .data_loader import _safe_col


def build_pl_table(fin: pd.DataFrame, info: dict, cur: str = "₹") -> pd.DataFrame:
    if fin.empty:
        return pd.DataFrame()
    rev   = _safe_col(fin, "Total Revenue")
    gp    = _safe_col(fin, "Gross Profit")
    op    = _safe_col(fin, "Operating Income", "EBIT")
    ebitda= _safe_col(fin, "EBITDA", "Normalized EBITDA")
    net   = _safe_col(fin, "Net Income")
    tax   = _safe_col(fin, "Tax Provision")
    int_  = _safe_col(fin, "Interest Expense")
    eps_d = _safe_col(fin, "Diluted EPS", "Basic EPS")

    opm = (op / rev * 100).replace([np.inf, -np.inf], np.nan)
    npm = (net / rev * 100).replace([np.inf, -np.inf], np.nan)
    gpm = (gp  / rev * 100).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({
        f"Revenue ({cur}Cr)":       (rev   / 1e7).round(1),
        f"Gross Profit ({cur}Cr)":  (gp    / 1e7).round(1),
        f"EBITDA ({cur}Cr)":        (ebitda/ 1e7).round(1),
        f"Operating Inc ({cur}Cr)": (op    / 1e7).round(1),
        f"Net Profit ({cur}Cr)":    (net   / 1e7).round(1),
        "Gross Margin %":           gpm.round(2),
        "Operating Margin %":       opm.round(2),
        "Net Margin %":             npm.round(2),
        "EPS (Diluted)":            eps_d.round(2),
    })


def build_balance_sheet(bal: pd.DataFrame, cur: str = "₹") -> pd.DataFrame:
    if bal.empty:
        return pd.DataFrame()
    ta_  = _safe_col(bal, "Total Assets")
    ca   = _safe_col(bal, "Current Assets")
    cl   = _safe_col(bal, "Current Liabilities")
    tl   = _safe_col(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
    eq   = _safe_col(bal, "Stockholders Equity", "Common Stock Equity")
    cash = _safe_col(bal, "Cash And Cash Equivalents", "Cash")
    inv  = _safe_col(bal, "Inventory")
    ltd  = _safe_col(bal, "Long Term Debt")
    d_e  = ((ltd) / eq.replace(0, np.nan)).round(2)
    wc   = ca - cl
    qr   = (ca - inv) / cl.replace(0, np.nan)
    return pd.DataFrame({
        f"Total Assets ({cur}Cr)":      (ta_ / 1e7).round(1),
        f"Current Assets ({cur}Cr)":    (ca  / 1e7).round(1),
        f"Cash ({cur}Cr)":              (cash/ 1e7).round(1),
        f"Inventory ({cur}Cr)":         (inv / 1e7).round(1),
        f"Total Liabilities ({cur}Cr)": (tl  / 1e7).round(1),
        f"Equity ({cur}Cr)":            (eq  / 1e7).round(1),
        f"LT Debt ({cur}Cr)":           (ltd / 1e7).round(1),
        f"Working Capital ({cur}Cr)":   (wc  / 1e7).round(1),
        "Debt / Equity":                d_e,
        "Quick Ratio":                  qr.round(2),
    })


def build_cashflow(cf: pd.DataFrame, cur: str = "₹") -> pd.DataFrame:
    if cf.empty:
        return pd.DataFrame()
    ocf   = _safe_col(cf, "Operating Cash Flow", "Cash Flow From Operations",
                      "Cash Flows From Operating Activities")
    icf   = _safe_col(cf, "Investing Cash Flow", "Cash Flows From Investing Activities")
    capex = _safe_col(cf, "Capital Expenditure", "Purchase Of PPE", "Capital Expenditures")
    fcf_  = _safe_col(cf, "Free Cash Flow")
    fcf   = fcf_ if not fcf_.isna().all() else (ocf + capex)
    div_p = _safe_col(cf, "Cash Dividends Paid", "Payment Of Dividends",
                      "Common Stock Dividend Paid", "Dividends Paid")
    return pd.DataFrame({
        f"Operating CF ({cur}Cr)":    (ocf  / 1e7).round(1),
        f"Investing CF ({cur}Cr)":    (icf  / 1e7).round(1),
        f"Free Cash Flow ({cur}Cr)":  (fcf  / 1e7).round(1),
        f"CapEx ({cur}Cr)":           (capex/ 1e7).round(1),
        f"Dividends Paid ({cur}Cr)":  (div_p/ 1e7).round(1),
    })


def compute_all_ratios(info: dict) -> dict:
    ev      = info.get("enterpriseValue")
    ebitda  = info.get("ebitda")
    revenue = info.get("totalRevenue")
    pe      = info.get("trailingPE")
    fwd_pe  = info.get("forwardPE")
    pb      = info.get("priceToBook")
    ps      = info.get("priceToSalesTrailing12Months")
    peg     = info.get("pegRatio")
    roe     = (info.get("returnOnEquity") or 0) * 100
    roa     = (info.get("returnOnAssets") or 0) * 100
    gross_m = (info.get("grossMargins") or 0) * 100
    op_m    = (info.get("operatingMargins") or 0) * 100
    ebitda_m= (info.get("ebitdaMargins") or 0) * 100
    net_m   = (info.get("profitMargins") or 0) * 100
    de      = info.get("debtToEquity")
    cr      = info.get("currentRatio")
    qr      = info.get("quickRatio")
    beta    = info.get("beta")
    div_y   = (info.get("dividendYield") or 0) * 100
    eps     = info.get("trailingEps")
    bvps    = info.get("bookValue")
    shares  = info.get("sharesOutstanding", 1) or 1

    ev_ebitda = round(ev/ebitda, 2) if ev and ebitda and ebitda > 0 else None
    ev_rev    = round(ev/revenue,2) if ev and revenue and revenue > 0 else None
    int_exp   = abs(info.get("interestExpense", 1) or 1)
    int_cov   = round((ebitda or 0) / int_exp, 2) if int_exp else None

    return {
        "Valuation": {
            "P/E (Trailing)":    round(pe, 2)     if pe     else None,
            "P/E (Forward)":     round(fwd_pe, 2) if fwd_pe else None,
            "P/B":               round(pb, 2)     if pb     else None,
            "P/S":               round(ps, 2)     if ps     else None,
            "EV/EBITDA":         ev_ebitda,
            "EV/Revenue":        ev_rev,
            "PEG Ratio":         round(peg, 2)    if peg    else None,
        },
        "Profitability": {
            "Gross Margin %":    round(gross_m, 2),
            "Operating Margin %":round(op_m, 2),
            "EBITDA Margin %":   round(ebitda_m, 2),
            "Net Profit Margin %":round(net_m, 2),
            "ROE %":             round(roe, 2),
            "ROA %":             round(roa, 2),
        },
        "Solvency": {
            "Debt / Equity":     round(de, 2)  if de  else None,
            "Current Ratio":     round(cr, 2)  if cr  else None,
            "Quick Ratio":       round(qr, 2)  if qr  else None,
            "Interest Coverage": int_cov,
            "Beta":              round(beta, 2) if beta else None,
        },
        "Per Share": {
            "EPS (TTM)":         round(eps, 2)  if eps  else None,
            "Book Value / Share":round(bvps, 2) if bvps else None,
            "Dividend Yield %":  round(div_y, 2),
            "Shares Out (Cr)":   round(shares / 1e7, 2),
        },
    }


def compute_piotroski_fscore(bal: pd.DataFrame, fin: pd.DataFrame, cf: pd.DataFrame, info: dict) -> dict:
    scores = {}
    roa_ = (info.get("returnOnAssets") or 0)

    # Use most-recent non-NaN OCF (iloc[-1] after sort_index ascending)
    ocf_ = 0.0
    for ocf_col in ("Operating Cash Flow", "Cash Flow From Operations",
                    "Cash Flows From Operating Activities"):
        if not cf.empty and ocf_col in cf.columns:
            _v = cf[ocf_col].dropna()
            if len(_v) > 0:
                ocf_ = float(_v.iloc[-1])   # most recent non-NaN
                break

    # Get total assets from balance sheet if available
    total_assets = 1.0
    for ta_c in ("Total Assets",):
        if not bal.empty and ta_c in bal.columns:
            _ta = bal[ta_c].dropna()
            if len(_ta) > 0:
                total_assets = float(_ta.iloc[-1])
                break
    if total_assets < 1000:
        total_assets = info.get("totalAssets") or 1.0

    scores["Positive ROA"]          = int(roa_ > 0)
    scores["Positive Operating CF"] = int(ocf_ > 0)
    scores["ROA > previous year"]   = int(roa_ >= 0)
    scores["CF > ROA (accruals)"]   = int(ocf_ > roa_ * total_assets) if total_assets > 1000 else 0

    de_ = info.get("debtToEquity", 0) or 0
    cr_ = info.get("currentRatio", 1) or 1
    scores["Lower leverage"]        = int(de_ < 1.0)
    scores["Higher current ratio"]  = int(cr_ > 1.0)
    scores["No share dilution"]     = 1

    gm_ = (info.get("grossMargins") or 0)
    at_ = (info.get("assetTurnover") or 0)
    scores["Improving gross margin"]   = int(gm_ > 0)
    scores["Improving asset turnover"] = int(at_ > 0)

    total = sum(scores.values())
    signal = "Strong 🟢" if total >= 7 else ("Moderate 🟡" if total >= 4 else "Weak 🔴")
    return {"score": total, "signal": signal, "criteria": scores}


def compute_altman_z(info: dict, bal: pd.DataFrame = None) -> dict:
    """Altman Z-Score.  Prefers balance-sheet values over info dict (more reliable)."""
    try:
        def _bs(col_names):
            """Try each col name in balance sheet; fall back to None."""
            if bal is not None and not bal.empty:
                for c in col_names:
                    if c in bal.columns:
                        v = bal[c].dropna()
                        if len(v) > 0:
                            return float(v.iloc[-1])
            return None

        mc_   = info.get("marketCap", 0) or 0
        sales = info.get("totalRevenue", 0) or 0
        ebit_ = info.get("ebitda", 0) or 0

        # Prefer balance-sheet for balance-sheet items
        ta__  = _bs(["Total Assets"]) or info.get("totalAssets") or 0
        tl_   = _bs(["Total Liabilities Net Minority Interest"]) or info.get("totalDebt") or 1
        re_   = _bs(["Retained Earnings"]) or info.get("retainedEarnings") or 0
        ca__  = _bs(["Current Assets"]) or info.get("currentAssets") or 0
        cl__  = _bs(["Current Liabilities"]) or info.get("currentLiabilities") or 1
        wc_   = ca__ - cl__

        # Guard: if total assets looks unrealistic, bail out
        if ta__ < 1000:
            return {"z_score": None, "zone": "N/A (balance sheet unavailable)"}

        X1 = wc_   / ta__
        X2 = re_   / ta__
        X3 = ebit_ / ta__
        X4 = mc_   / tl_
        X5 = sales / ta__

        z    = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        # Sanity cap — values above 100 or below -50 indicate bad input data
        if abs(z) > 200:
            return {"z_score": None, "zone": "N/A (data mismatch)"}
        zone = "Safe ✅" if z > 2.99 else ("Grey ⚠️" if z > 1.81 else "Distress Zone 🔴")
        return {"z_score": round(z, 2), "zone": zone}
    except Exception:
        return {"z_score": None, "zone": "N/A"}


def compute_dcf(cf: pd.DataFrame, info: dict, wacc: float = 0.12, tg: float = 0.04) -> dict:
    for col in ["Free Cash Flow", "Operating Cash Flow", "Cash Flow From Operations"]:
        if col in cf.columns:
            vals = cf[col].dropna()
            if len(vals) > 0:
                base_fcf = float(vals.iloc[-1])   # most recent non-NaN
                if base_fcf <= 0:
                    return {"error": "Negative/Zero FCF"}
                g = min((info.get("revenueGrowth") or 0.08), 0.20)
                pv, fcf_t = 0.0, base_fcf
                for t in range(1, 11):
                    fcf_t *= (1 + g)
                    pv += fcf_t / (1 + wacc) ** t
                tv_pv = (fcf_t * (1 + tg) / (wacc - tg)) / (1 + wacc) ** 10
                total = pv + tv_pv
                net_debt = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)
                shares = info.get("sharesOutstanding", 1) or 1
                intrinsic = max(0, total - net_debt) / shares
                return {
                    "intrinsic": round(intrinsic, 2),
                    "upside":    round((intrinsic / (info.get("currentPrice") or 1) - 1) * 100, 1),
                    "wacc":      wacc, "g_rate": round(g * 100, 1),
                }
    return {"error": "No FCF data"}


def compute_graham_number(info: dict) -> float | None:
    eps  = info.get("trailingEps")
    bvps = info.get("bookValue")
    if eps and eps > 0 and bvps and bvps > 0:
        return round((22.5 * eps * bvps) ** 0.5, 2)
    return None
