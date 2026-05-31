"""
SCREENER — Deep Fundamental + Technical Stock Analysis
Streamlit app inspired by screener.in × groww.in
"""
import warnings; warnings.filterwarnings("ignore")

import os, io, zipfile, json, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv, find_dotenv, set_key
from datetime import datetime

from modules.data_loader   import download_full_data, download_benchmark, compute_technical_indicators, _safe_col
from modules.fundamentals  import (build_pl_table, build_balance_sheet, build_cashflow,
                                    compute_all_ratios, compute_piotroski_fscore,
                                    compute_altman_z, compute_dcf, compute_graham_number)
from modules.peers         import get_peer_data
from modules.screener      import DEFAULT_UNIVERSE, fetch_screener_data, apply_filters

# ══════════════════════════════════════════════════════════════════════════════
#  Bundle helpers — save / restore full analysis as a ZIP file
# ══════════════════════════════════════════════════════════════════════════════

def _info_json_safe(info: dict) -> dict:
    """Return a copy of the yfinance info dict filtered to JSON-serialisable primitives."""
    out: dict = {}
    for k, v in info.items():
        if v is None:
            out[k] = None
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int):
            out[k] = v
        elif isinstance(v, float):
            out[k] = None if (math.isnan(v) or math.isinf(v)) else v
        elif isinstance(v, str):
            out[k] = v
        else:
            try:
                if isinstance(v, np.integer):
                    out[k] = int(v)
                elif isinstance(v, np.floating):
                    out[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                elif isinstance(v, np.bool_):
                    out[k] = bool(v)
                # lists, dicts, and other complex types are intentionally skipped
            except Exception:
                pass
    return out


def _build_screener_bundle() -> bytes:
    """Serialise all screener analysis results to an in-memory ZIP bundle."""
    raw         = st.session_state.get("raw", {})
    df_ta       = st.session_state.get("df_ta", pd.DataFrame())
    bench_hist  = st.session_state.get("bench_hist", pd.DataFrame())
    peer_df     = st.session_state.get("peer_df", pd.DataFrame())
    peer_list   = st.session_state.get("peer_list", [])
    peer_method = st.session_state.get("peer_method", "")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if not df_ta.empty:
            zf.writestr("df_ta.csv", df_ta.to_csv())
        if not bench_hist.empty:
            zf.writestr("bench_hist.csv", bench_hist.to_csv())
        for key, fname in [("fin", "fin.csv"), ("bal", "bal.csv"),
                            ("cf", "cf.csv"),  ("qfin", "qfin.csv")]:
            df_ = raw.get(key, pd.DataFrame())
            if df_ is not None and not df_.empty:
                zf.writestr(fname, df_.to_csv())
        if not peer_df.empty:
            zf.writestr("peers.csv", peer_df.to_csv())
        holders = raw.get("holders_inst")
        if holders is not None and not holders.empty:
            zf.writestr("holders_inst.csv", holders.to_csv(index=False))
        info_safe = _info_json_safe(raw.get("info") or {})
        meta = {
            "ticker":       st.session_state.get("_screener_ticker", ""),
            "period":       st.session_state.get("_screener_period", "5y"),
            "bench":        st.session_state.get("_screener_bench",  "^NSEI"),
            "company_name": raw.get("company_name", ""),
            "sector":       raw.get("sector", ""),
            "industry":     raw.get("industry", ""),
            "exchange":     raw.get("exchange", ""),
            "currency":     raw.get("currency", "INR"),
            "currency_sym": raw.get("currency_sym", "₹"),
            "peer_list":    peer_list,
            "peer_method":  peer_method,
        }
        zf.writestr("session_data.json",
                    json.dumps({"info": info_safe, "meta": meta}))
    buf.seek(0)
    return buf.read()


def _restore_screener_bundle(uploaded_file) -> bool:
    """Restore all screener analysis results from a bundle ZIP into session_state."""
    try:
        with st.spinner("⚡ Loading analysis bundle…"):
            raw_bytes = uploaded_file.read()
            buf = io.BytesIO(raw_bytes)
            with zipfile.ZipFile(buf, "r") as zf:
                names = set(zf.namelist())
                payload = json.loads(zf.read("session_data.json").decode("utf-8"))
                info    = payload["info"]
                meta    = payload["meta"]

                def _load_df_csv(fname: str, parse_dates: bool = True) -> pd.DataFrame:
                    if fname not in names:
                        return pd.DataFrame()
                    try:
                        df_ = pd.read_csv(
                            io.BytesIO(zf.read(fname)),
                            index_col=0,
                        )
                        if parse_dates:
                            df_.index = pd.to_datetime(df_.index, errors="coerce")
                        return df_
                    except Exception:
                        return pd.DataFrame()

                df_ta      = _load_df_csv("df_ta.csv")
                bench_hist = _load_df_csv("bench_hist.csv")
                fin        = _load_df_csv("fin.csv")
                bal        = _load_df_csv("bal.csv")
                cf_df      = _load_df_csv("cf.csv")
                qfin       = _load_df_csv("qfin.csv")
                peer_df    = _load_df_csv("peers.csv", parse_dates=False)
                if "holders_inst.csv" in names:
                    try:
                        holders_inst = pd.read_csv(
                            io.BytesIO(zf.read("holders_inst.csv")))
                    except Exception:
                        holders_inst = pd.DataFrame()
                else:
                    holders_inst = pd.DataFrame()

                # Reconstruct hist from df_ta OHLCV columns
                hist_cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                                         "Dividends", "Stock Splits"]
                             if c in df_ta.columns]
                hist = df_ta[hist_cols].copy() if hist_cols else df_ta.iloc[:, :5].copy()

                raw = {
                    "info":          info,
                    "hist":          hist,
                    "fin":           fin,
                    "bal":           bal,
                    "cf":            cf_df,
                    "qfin":          qfin,
                    "qbal":          pd.DataFrame(),
                    "qcf":           pd.DataFrame(),
                    "actions":       pd.DataFrame(),
                    "holders_inst":  holders_inst,
                    "holders_major": pd.DataFrame(),
                    "calendar":      None,
                    "company_name":  meta.get("company_name", ""),
                    "sector":        meta.get("sector", ""),
                    "industry":      meta.get("industry", ""),
                    "exchange":      meta.get("exchange", ""),
                    "currency":      meta.get("currency", "INR"),
                    "currency_sym":  meta.get("currency_sym", "₹"),
                }
                st.session_state["raw"]                    = raw
                st.session_state["df_ta"]                  = df_ta
                st.session_state["bench_hist"]             = bench_hist
                st.session_state["peer_df"]                = peer_df
                st.session_state["peer_list"]              = meta.get("peer_list", [])
                st.session_state["peer_method"]            = meta.get("peer_method", "bundle")
                st.session_state["_screener_ticker"]       = meta.get("ticker", "")
                st.session_state["_screener_period"]       = meta.get("period", "5y")
                st.session_state["_screener_bench"]        = meta.get("bench",  "^NSEI")
                st.session_state["loaded_screener_ticker"] = meta.get("ticker", "")
                st.session_state["_screener_loaded"]       = True
                return True
    except Exception as e:
        st.error(f"❌ Failed to load bundle: {e}")
        return False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCREENER — Deep Stock Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme override ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0d1117; color: #c9d1d9; }
  .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
  [data-testid="stSidebar"] { background: #161b22; }
  .block-container { padding-top: 1rem; }
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=55)
    st.title("SCREENER")
    st.caption("screener.in × groww.in — Python Edition")
    st.divider()

    st.subheader("📈 Stock")
    ticker  = st.text_input("Ticker Symbol", value="HAL.NS",
                              help="NSE: TICKER.NS | BSE: TICKER.BO | US: AAPL")
    period  = st.selectbox("Historical Period", ["1y","2y","3y","5y","10y","max"], index=3)
    bench   = st.text_input("Benchmark", value="^NSEI",
                              help="^NSEI (Nifty), ^GSPC (S&P 500)")

    st.subheader("🔑 API Keys (optional)")
    load_dotenv()
    def _k(lbl, env):
        return st.text_input(lbl, value=os.getenv(env,""), type="password", key=env)

    groq_key   = _k("Groq (peer discovery)", "GROQ_API_KEY")
    gemini_key = _k("Gemini (peer discovery)", "GEMINI_API_KEY")

    save_env = st.checkbox("💾 Save keys to .env")
    if save_env and st.button("Save"):
        env_file = find_dotenv() or os.path.join(os.path.dirname(__file__), ".env")
        for k, v in [("GROQ_API_KEY", groq_key), ("GEMINI_API_KEY", gemini_key)]:
            if v: set_key(env_file, k, v)
        st.success("Keys saved")

    st.divider()
    run_btn = st.button("🔍 Analyse Stock", use_container_width=True, type="primary")
    st.divider()
    st.subheader("📂 Load Previous Analysis")
    st.caption("Upload a bundle ZIP to restore results instantly — no re-downloading")
    uploaded_bundle = st.file_uploader(
        "Upload Bundle ZIP", type=["zip"], key="screener_bundle_uploader"
    )
    load_btn = st.button(
        "⚡ Load from Bundle",
        use_container_width=True,
        disabled=uploaded_bundle is None,
        key="screener_load_btn",
    )
    st.divider()
    st.caption("ℹ️ Data from Yahoo Finance via yfinance")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 SCREENER — Deep Fundamental + Technical Analysis")
st.caption("Inspired by screener.in × groww.in  ·  All data via yfinance")

# ── Bundle load trigger (runs before early-stop so it works on first visit too) ──
if load_btn and uploaded_bundle is not None and not run_btn:
    ok = _restore_screener_bundle(uploaded_bundle)
    if ok:
        st.rerun()
    st.stop()

if not run_btn and "raw" not in st.session_state:
    st.info("Enter a ticker symbol and click **Analyse Stock** to begin, "
            "or upload a bundle ZIP via the sidebar.")
    st.stop()

# ── Data load (on button press) ───────────────────────────────────────────────
if run_btn:
    st.session_state.clear()

ticker_key = ticker.strip().upper()
# Override with bundle ticker when analysis was restored from a ZIP
if st.session_state.get("_screener_loaded"):
    ticker_key = st.session_state.get("loaded_screener_ticker", ticker_key)

if run_btn or "raw" not in st.session_state:
    with st.spinner("⬇️  Downloading data…"):
        raw = download_full_data(ticker_key, period)
        st.session_state["raw"]               = raw
        st.session_state["df_ta"]             = compute_technical_indicators(raw["hist"])
        bench_hist = download_benchmark(bench, period)
        st.session_state["bench_hist"]        = bench_hist
        st.session_state["_screener_ticker"]  = ticker_key
        st.session_state["_screener_period"]  = period
        st.session_state["_screener_bench"]   = bench

raw       = st.session_state["raw"]
df_ta     = st.session_state["df_ta"]
bench_hist= st.session_state["bench_hist"]

info         = raw["info"]
company_name = raw["company_name"]
sector       = raw["sector"]
cur          = raw["currency_sym"]
exchange     = raw["exchange"]
hist         = raw["hist"]

price    = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
prev_cl  = info.get("regularMarketPreviousClose") or float(hist["Close"].iloc[-2])
chg_pct  = (price - prev_cl) / prev_cl * 100

# ── Header ─────────────────────────────────────────────────────────────────────
sig_color = "#4caf50" if chg_pct >= 0 else "#f44336"
c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
with c1:
    st.subheader(f"{company_name}  ({ticker_key})")
    st.caption(f"{sector}  ·  {raw['industry']}  ·  {exchange}")
with c2:
    st.metric("Price", f"{cur}{price:,.2f}", f"{chg_pct:+.2f}%",
               delta_color="normal" if chg_pct >= 0 else "inverse")
with c3:
    mc = info.get("marketCap")
    st.metric("Market Cap", f"{cur}{mc/1e7:,.0f} Cr" if mc else "N/A")
with c4:
    pe = info.get("trailingPE")
    st.metric("P/E", f"{pe:.1f}" if pe else "N/A")

# ── Download buttons ─────────────────────────────────────────────────────────
_ts = pd.Timestamp.now().strftime('%Y%m%d')
dbc1, dbc2 = st.columns(2)
with dbc1:
    bundle_bytes = _build_screener_bundle()
    st.download_button(
        "📦 Download Analysis Bundle ZIP",
        bundle_bytes,
        file_name=f"SCREENER_{ticker_key}_{_ts}_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        help="Save bundle to reload later — peers, financials, indicators all included.",
    )
with dbc2:
    csv_bytes = df_ta.to_csv().encode()
    st.download_button(
        "⬇ Price + Indicators CSV",
        csv_bytes,
        file_name=f"{ticker_key}_indicators.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📉 Price Chart",
    "💹 Financials",
    "📐 Ratios",
    "🗓️ Quarterly",
    "🥧 Shareholding",
    "👥 Peers",
    "💰 Valuation",
    "📊 Technical",
    "🔍 Screener",
    "🧮 Calculator",
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 0 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Company Overview")

    # Key metrics grid (groww.in style)
    m_cols = st.columns(4)
    hi52 = info.get("fiftyTwoWeekHigh")
    lo52 = info.get("fiftyTwoWeekLow")
    book_val = info.get("bookValue")
    div_yield = (info.get("dividendYield") or 0) * 100
    roe  = (info.get("returnOnEquity") or 0) * 100
    roa  = (info.get("returnOnAssets") or 0) * 100
    beta = info.get("beta")
    pb   = info.get("priceToBook")
    fwd_pe = info.get("forwardPE")
    de   = info.get("debtToEquity")

    kv = [
        ("52W High", f"{cur}{hi52:,.2f}" if hi52 else "N/A"),
        ("52W Low",  f"{cur}{lo52:,.2f}" if lo52 else "N/A"),
        ("P/B",      f"{pb:.2f}" if pb else "N/A"),
        ("EPS (TTM)",f"{cur}{info.get('trailingEps'):,.2f}" if info.get("trailingEps") else "N/A"),
        ("ROE",      f"{roe:.2f}%"),
        ("ROA",      f"{roa:.2f}%"),
        ("Div Yield",f"{div_yield:.2f}%"),
        ("D/E Ratio",f"{de:.2f}" if de else "N/A"),
        ("Beta",     f"{beta:.2f}" if beta else "N/A"),
        ("Fwd P/E",  f"{fwd_pe:.2f}" if fwd_pe else "N/A"),
        ("Book Value",f"{cur}{book_val:,.2f}" if book_val else "N/A"),
        ("Exchange", exchange),
    ]
    for i, (label, value) in enumerate(kv):
        m_cols[i % 4].metric(label, value)

    # 52-week range bar
    if hi52 and lo52:
        pct_pos = (price - lo52) / (hi52 - lo52) if (hi52 - lo52) > 0 else 0.5
        st.progress(min(1.0, pct_pos),
                     text=f"52-week range: {cur}{lo52:,.2f} ◄──● {pct_pos*100:.0f}% from low ──► {cur}{hi52:,.2f}")

    st.divider()

    # Pros & Cons
    col_pros, col_cons = st.columns(2)
    pros, cons = [], []
    if pe and pe < 20: pros.append(f"✅ Reasonable P/E of {pe:.1f}x")
    elif pe and pe > 40: cons.append(f"⚠️ Expensive P/E of {pe:.1f}x")
    if pb and pb < 1: pros.append(f"✅ Trading below book value (P/B={pb:.2f})")
    if roe > 20: pros.append(f"✅ Strong ROE of {roe:.1f}%")
    elif roe < 8 and roe > 0: cons.append(f"⚠️ Low ROE of {roe:.1f}%")
    nm_ = (info.get("profitMargins") or 0) * 100
    if nm_ > 15: pros.append(f"✅ Healthy net margin {nm_:.1f}%")
    elif nm_ < 5 and nm_ != 0: cons.append(f"⚠️ Thin net margin {nm_:.1f}%")
    if de and de < 0.5: pros.append(f"✅ Low D/E of {de:.2f}")
    elif de and de > 2: cons.append(f"⚠️ High D/E of {de:.2f}")
    tc = info.get("totalCash", 0) or 0
    td = info.get("totalDebt", 0) or 0
    if tc > td: pros.append("✅ Cash exceeds total debt")
    rg = (info.get("revenueGrowth") or 0) * 100
    if rg > 15: pros.append(f"✅ Revenue growing {rg:.1f}%")
    elif rg < 0: cons.append(f"⚠️ Declining revenue ({rg:.1f}%)")
    if div_yield > 2: pros.append(f"✅ Dividend yield {div_yield:.2f}%")
    cr_v = info.get("currentRatio")
    if cr_v and cr_v > 2: pros.append(f"✅ Strong liquidity (CR={cr_v:.1f})")
    elif cr_v and cr_v < 1: cons.append(f"⚠️ Weak liquidity (CR={cr_v:.1f})")

    with col_pros:
        st.markdown("**🟢 Strengths**")
        for p in pros: st.markdown(p)
        if not pros: st.caption("—")
    with col_cons:
        st.markdown("**🔴 Concerns**")
        for c_ in cons: st.markdown(c_)
        if not cons: st.success("No major concerns found.")

    # Company about
    with st.expander("ℹ️ About the Company"):
        desc = info.get("longBusinessSummary", "No description available.")
        emp  = info.get("fullTimeEmployees")
        web  = info.get("website","N/A")
        hq   = f"{info.get('city','')}, {info.get('country','')}".strip(", ")
        st.markdown(f"**HQ:** {hq}  |  **Employees:** {f'{emp:,}' if emp else 'N/A'}  |  **Website:** [{web}]({web})")
        st.write(desc[:1200] + ("…" if len(desc) > 1200 else ""))


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PRICE CHART
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Interactive Price Chart")

    chart_type = st.radio("Chart type", ["Candlestick", "Line", "OHLC"], horizontal=True)
    show_ma    = st.multiselect("Moving Averages", ["SMA20","SMA50","SMA200","EMA20","EMA50"], default=["SMA50","SMA200"])
    range_sel  = st.selectbox("Range", ["1M","3M","6M","1Y","2Y","5Y","All"], index=3)

    range_map  = {"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504,"5Y":1260,"All":len(df_ta)}
    n_bars     = min(range_map.get(range_sel, 252), len(df_ta))
    d          = df_ta.tail(n_bars).copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.72, 0.28], vertical_spacing=0.03)

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"],
            low=d["Low"], close=d["Close"],
            increasing_line_color="#4caf50", decreasing_line_color="#f44336",
            name="Price"), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                                  line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    else:
        fig.add_trace(go.Ohlc(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name="OHLC"), row=1, col=1)

    ma_colors = {"SMA20":"#f9a825","SMA50":"#58a6ff","SMA200":"#ce93d8","EMA20":"#ff7043","EMA50":"#80cbc4"}
    for ma in show_ma:
        if ma in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=ma,
                                      line=dict(color=ma_colors.get(ma,"#fff"), width=1.2)), row=1, col=1)

    vol_colors = ["#4caf50" if c >= o else "#f44336" for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors, name="Vol", opacity=0.6), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=580, paper_bgcolor="#0d1117",
                       xaxis_rangeslider_visible=False, showlegend=True,
                       legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Returns comparison
    if not bench_hist.empty:
        periods_r = {"1M":21,"3M":63,"6M":126,"1Y":252,"3Y":756}
        sc, bc = hist["Close"], bench_hist["Close"]
        ret_data = {}
        for lbl, days in periods_r.items():
            if len(sc) >= days:
                ret_data[lbl] = {"Stock": round((sc.iloc[-1]/sc.iloc[-days]-1)*100, 2)}
            if len(bc) >= days:
                ret_data.setdefault(lbl, {})["Benchmark"] = round((bc.iloc[-1]/bc.iloc[-days]-1)*100, 2)
        df_ret = pd.DataFrame(ret_data).T.dropna(how="all")
        if not df_ret.empty:
            fig_ret = go.Figure()
            for col, color in [("Stock","#58a6ff"), ("Benchmark","#f9a825")]:
                if col in df_ret.columns:
                    fig_ret.add_trace(go.Bar(name=col, x=df_ret.index, y=df_ret[col], marker_color=color))
            fig_ret.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.3)
            fig_ret.update_layout(template="plotly_dark", barmode="group", height=300,
                                   title="Period Returns: Stock vs Benchmark (%)", paper_bgcolor="#0d1117")
            st.plotly_chart(fig_ret, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FINANCIALS (P&L / BS / CF)
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    fin_tab = st.tabs(["📈 P&L Statement", "🏦 Balance Sheet", "💵 Cash Flow"])

    with fin_tab[0]:
        st.subheader("Annual Profit & Loss")
        pl_df = build_pl_table(raw["fin"], info, cur)
        if not pl_df.empty:
            yrs   = [str(i.year) for i in pl_df.index]
            style_cols = [c for c in ["Operating Margin %","Net Margin %","Gross Margin %"] if c in pl_df.columns]
            st.dataframe(pl_df.style.background_gradient(subset=style_cols, cmap="RdYlGn", vmin=-5, vmax=35)
                         .format(na_rep="—"), use_container_width=True)
            # Revenue + Profit bar
            rev_col = [c for c in pl_df.columns if "Revenue" in c]
            net_col = [c for c in pl_df.columns if "Net Profit" in c]
            if rev_col and net_col:
                fig_pl = make_subplots(rows=1, cols=2,
                                        subplot_titles=["Revenue & Net Profit", "Margin Trends"])
                fig_pl.add_trace(go.Bar(name="Revenue",    x=yrs, y=pl_df[rev_col[0]], marker_color="#58a6ff"), row=1, col=1)
                fig_pl.add_trace(go.Bar(name="Net Profit", x=yrs, y=pl_df[net_col[0]], marker_color="#4caf50"), row=1, col=1)
                for mg, cl in [("Operating Margin %","#f9a825"),("Net Margin %","#ce93d8")]:
                    if mg in pl_df.columns:
                        fig_pl.add_trace(go.Scatter(name=mg, x=yrs, y=pl_df[mg],
                                                     mode="lines+markers", line=dict(color=cl, width=2)), row=1, col=2)
                fig_pl.update_layout(template="plotly_dark", height=380, barmode="group", paper_bgcolor="#0d1117")
                st.plotly_chart(fig_pl, use_container_width=True)
        else:
            st.warning("Annual P&L data not available.")

    with fin_tab[1]:
        st.subheader("Annual Balance Sheet")
        bs_df = build_balance_sheet(raw["bal"], cur)
        if not bs_df.empty:
            yrs = [str(i.year) for i in bs_df.index]
            de_col = [c for c in bs_df.columns if "Debt / Equity" in c or "Debt/Equity" in c]
            st.dataframe(bs_df.style.background_gradient(subset=de_col if de_col else [], cmap="RdYlGn_r", vmin=0, vmax=3)
                         .format(na_rep="—"), use_container_width=True)
            ta_col = [c for c in bs_df.columns if "Total Assets" in c]
            tl_col = [c for c in bs_df.columns if "Total Liabilities" in c]
            eq_col = [c for c in bs_df.columns if "Equity" in c and "Debt" not in c]
            if ta_col and tl_col:
                fig_bs = go.Figure()
                for col, color in [(ta_col[0],"#58a6ff"),(tl_col[0],"#f44336"),(eq_col[0] if eq_col else None,"#4caf50")]:
                    if col and col in bs_df.columns:
                        fig_bs.add_trace(go.Bar(name=col, x=yrs, y=bs_df[col], marker_color=color))
                fig_bs.update_layout(template="plotly_dark", barmode="group", height=340, paper_bgcolor="#0d1117",
                                      title="Assets vs Liabilities vs Equity")
                st.plotly_chart(fig_bs, use_container_width=True)
        else:
            st.warning("Balance sheet data not available.")

    with fin_tab[2]:
        st.subheader("Annual Cash Flow")
        cf_df = build_cashflow(raw["cf"], cur)
        if not cf_df.empty:
            yrs = [str(i.year) for i in cf_df.index]
            st.dataframe(cf_df.style.format(na_rep="—"), use_container_width=True)
            fig_cf = go.Figure()
            colors_cf = ["#4caf50","#f44336","#58a6ff"]
            for i_c, col in enumerate(cf_df.columns[:3]):
                fig_cf.add_trace(go.Bar(name=col, x=yrs, y=cf_df[col], marker_color=colors_cf[i_c]))
            fig_cf.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.3)
            fig_cf.update_layout(template="plotly_dark", barmode="group", height=340,
                                  title="Cash Flow Trends", paper_bgcolor="#0d1117")
            st.plotly_chart(fig_cf, use_container_width=True)
        else:
            st.warning("Cash flow data not available.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — RATIOS
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Financial Ratios")
    all_ratios = compute_all_ratios(info)

    r_cols = st.columns(2)
    for i_r, (cat, ratios) in enumerate(all_ratios.items()):
        with r_cols[i_r % 2]:
            st.markdown(f"**{cat}**")
            df_r = pd.DataFrame.from_dict(ratios, orient="index", columns=["Value"])
            df_r.index.name = "Ratio"
            st.dataframe(df_r.style.format(na_rep="—"), use_container_width=True, height=240)

    # Radar chart
    st.subheader("Fundamental Scorecard")
    def _norm(v, lo, hi):
        if v is None: return 0
        try: return max(0, min(1, (float(v) - lo) / (hi - lo)))
        except: return 0

    cats   = ["ROE", "Net Margin", "Gross Margin", "Current Ratio", "Value (low P/E)", "Safety (low D/E)"]
    prof   = all_ratios["Profitability"]
    solv   = all_ratios["Solvency"]
    vals_r = [
        _norm(prof.get("ROE %"), 0, 40),
        _norm(prof.get("Net Profit Margin %"), 0, 30),
        _norm(prof.get("Gross Margin %"), 0, 60),
        _norm(solv.get("Current Ratio"), 0, 4),
        _norm(40 - (all_ratios["Valuation"].get("P/E (Trailing)") or 40), 0, 40),
        _norm(3 - (solv.get("Debt / Equity") or 3), 0, 3),
    ]
    fig_r = go.Figure(go.Scatterpolar(
        r=vals_r + [vals_r[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(88,166,255,0.2)",
        line=dict(color="#58a6ff", width=2)
    ))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1], showticklabels=False),
                                    bgcolor="#161b22"),
                         template="plotly_dark", height=400, paper_bgcolor="#0d1117",
                         title="Fundamental Scorecard (normalised)")
    st.plotly_chart(fig_r, use_container_width=True)

    # Health scores
    st.subheader("Financial Health Scores")
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.markdown("**🏥 Altman Z-Score**")
        az = compute_altman_z(info, raw["bal"])
        z_color = "#4caf50" if (az["z_score"] or 0) > 2.99 else ("#f9a825" if (az["z_score"] or 0) > 1.81 else "#f44336")
        st.markdown(f"<h2 style='color:{z_color};'>{az['z_score'] or 'N/A'}</h2>", unsafe_allow_html=True)
        st.caption(f"{az['zone']}  ·  >2.99 Safe  ·  1.81-2.99 Grey  ·  <1.81 Distress")
    with c_h2:
        st.markdown("**📋 Piotroski F-Score**")
        pio = compute_piotroski_fscore(raw["bal"], raw["fin"], raw["cf"], info)
        f_color = "#4caf50" if pio["score"] >= 7 else ("#f9a825" if pio["score"] >= 4 else "#f44336")
        st.markdown(f"<h2 style='color:{f_color};'>{pio['score']}/9</h2>", unsafe_allow_html=True)
        st.caption(f"{pio['signal']}  ·  7-9 Strong  ·  4-6 Moderate  ·  0-3 Weak")
        with st.expander("Criteria"):
            for k, v in pio["criteria"].items():
                st.write(f"{'✅' if v else '❌'}  {k}")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — QUARTERLY RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Quarterly Results")
    qfin = raw["qfin"]
    if not qfin.empty:
        q_rev  = _safe_col(qfin, "Total Revenue")
        q_op   = _safe_col(qfin, "Operating Income", "EBIT")
        q_net  = _safe_col(qfin, "Net Income")
        q_eps  = _safe_col(qfin, "Diluted EPS", "Basic EPS")
        q_opm  = (q_op / q_rev * 100).replace([np.inf, -np.inf], np.nan)
        q_npm  = (q_net/ q_rev * 100).replace([np.inf, -np.inf], np.nan)

        q_df = pd.DataFrame({
            f"Revenue ({cur}Cr)":    (q_rev / 1e7).round(1),
            f"Operating Inc ({cur}Cr)": (q_op / 1e7).round(1),
            f"Net Profit ({cur}Cr)": (q_net / 1e7).round(1),
            "OPM %":                 q_opm.round(2),
            "NPM %":                 q_npm.round(2),
            "EPS":                   q_eps.round(2),
        }).tail(12)

        st.dataframe(q_df.style
                     .background_gradient(subset=["OPM %","NPM %"], cmap="RdYlGn", vmin=-5, vmax=30)
                     .format(na_rep="—"), use_container_width=True)

        labels_q = [str(d.date()) for d in q_df.index]
        fig_q = make_subplots(rows=2, cols=2,
                               subplot_titles=["Quarterly Revenue","Net Profit","OPM %","EPS"],
                               vertical_spacing=0.14)
        for (val, r, c_, col) in [
            (q_df[f"Revenue ({cur}Cr)"],    1, 1, "#58a6ff"),
            (q_df[f"Net Profit ({cur}Cr)"], 1, 2, "#4caf50"),
            (q_df["OPM %"],                 2, 1, "#f9a825"),
            (q_df["EPS"],                   2, 2, "#ce93d8"),
        ]:
            bc = [col if v >= 0 else "#f44336" for v in val.fillna(0)]
            fig_q.add_trace(go.Bar(x=labels_q, y=val, marker_color=bc, showlegend=False), row=r, col=c_)
        fig_q.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0d1117")
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.warning("Quarterly data not available for this ticker.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SHAREHOLDING
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Shareholding Pattern")
    promoter = (info.get("heldPercentInsiders") or 0) * 100
    inst_pct = (info.get("heldPercentInstitutions") or 0) * 100
    public_p = max(0, 100 - promoter - inst_pct)

    labels_sh  = ["Promoter", "Institutional", "Public / Others"]
    values_sh  = [round(promoter,2), round(inst_pct,2), round(public_p,2)]
    colors_sh  = ["#58a6ff","#4caf50","#f9a825"]

    c_s1, c_s2 = st.columns([1, 2])
    with c_s1:
        fig_pie = go.Figure(go.Pie(
            labels=labels_sh, values=values_sh,
            marker=dict(colors=colors_sh),
            textinfo="label+percent", hole=0.4,
        ))
        fig_pie.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d1117",
                               showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_s2:
        df_sh = pd.DataFrame({"Category": labels_sh, "% Held": values_sh})
        st.dataframe(df_sh, use_container_width=True)
        st.caption("ℹ️ Detailed quarterly shareholding trends are available for Indian stocks via BSE/NSE disclosures. yfinance provides the latest snapshot only.")

    if raw["holders_inst"] is not None and not raw["holders_inst"].empty:
        with st.expander("📋 Top Institutional Holders"):
            st.dataframe(raw["holders_inst"].head(15), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — PEERS
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader(f"Peer Comparison — {sector}")
    if "peer_df" not in st.session_state:
        with st.spinner("Finding peers…"):
            peer_list, peer_df, peer_method = get_peer_data(
                ticker_key, sector,
                company_name=company_name, exchange=exchange,
                groq_key=groq_key, gemini_key=gemini_key,
            )
            st.session_state["peer_list"]   = peer_list
            st.session_state["peer_df"]     = peer_df
            st.session_state["peer_method"] = peer_method
    else:
        peer_list   = st.session_state["peer_list"]
        peer_df     = st.session_state["peer_df"]
        peer_method = st.session_state["peer_method"]

    method_badge = "🤖 LLM-discovered" if "LLM" in peer_method or "Groq" in peer_method or "Gemini" in peer_method else "📚 Sector database"
    st.caption(f"Peer discovery: **{method_badge}** via *{peer_method}*  ·  {len(peer_list)} companies")

    if not peer_df.empty:
        st.dataframe(peer_df, use_container_width=True)
        peer_tabs = st.tabs(["P/E","Market Cap","ROE vs P/B"])
        with peer_tabs[0]:
            if "P/E" in peer_df.columns:
                d_pe = peer_df["P/E"].dropna().reset_index()
                bc   = ["#f9a825" if t == ticker_key else "#58a6ff" for t in d_pe["Ticker"]]
                fig_pe= go.Figure(go.Bar(x=d_pe["Ticker"], y=d_pe["P/E"], marker_color=bc))
                fig_pe.update_layout(template="plotly_dark", height=300, title="P/E Comparison", paper_bgcolor="#0d1117")
                st.plotly_chart(fig_pe, use_container_width=True)
        with peer_tabs[1]:
            if "Mkt Cap (Cr)" in peer_df.columns:
                d_mc = peer_df["Mkt Cap (Cr)"].dropna().reset_index()
                fig_mc= go.Figure(go.Bar(x=d_mc["Ticker"], y=d_mc["Mkt Cap (Cr)"],
                                          marker_color=["#f9a825" if t==ticker_key else "#58a6ff" for t in d_mc["Ticker"]]))
                fig_mc.update_layout(template="plotly_dark", height=300, title="Market Cap (₹ Cr)", paper_bgcolor="#0d1117")
                st.plotly_chart(fig_mc, use_container_width=True)
        with peer_tabs[2]:
            if "ROE %" in peer_df.columns and "P/B" in peer_df.columns:
                d_rb = peer_df[["ROE %","P/B","Mkt Cap (Cr)"]].dropna().reset_index()
                fig_rb= px.scatter(d_rb, x="P/B", y="ROE %", text="Ticker",
                                    size="Mkt Cap (Cr)" if "Mkt Cap (Cr)" in d_rb else None,
                                    color="ROE %", color_continuous_scale="RdYlGn",
                                    title="ROE % vs P/B")
                fig_rb.update_traces(textposition="top center")
                fig_rb.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0d1117")
                st.plotly_chart(fig_rb, use_container_width=True)
    else:
        st.warning("Could not fetch peer data.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — VALUATION
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Valuation Models")
    v_cols = st.columns(3)

    # Pre-initialise so the summary table never has NameError
    graham = None
    g_up   = 0.0
    pe_fair= None
    pe_up  = 0.0
    dcf_res= {"error": "Not computed"}

    # DCF
    with v_cols[0]:
        st.markdown("**💡 DCF (10-Year)**")
        wacc_in = st.slider("WACC %", 8, 20, 12, key="wacc") / 100
        tg_in   = st.slider("Terminal Growth %", 2, 8, 4, key="tg") / 100
        dcf_res = compute_dcf(raw["cf"], info, wacc=wacc_in, tg=tg_in)
        if not dcf_res.get("error"):
            intr   = dcf_res["intrinsic"]
            upside = dcf_res["upside"]
            d_col  = "#4caf50" if upside and upside > 0 else "#f44336"
            st.markdown(f"<h3 style='color:{d_col};'>{cur}{intr:,.2f}</h3>", unsafe_allow_html=True)
            st.metric("Upside", f"{upside:+.1f}%" if upside else "N/A")
            st.caption(f"WACC={wacc_in*100:.0f}%  Growth={dcf_res['g_rate']}%  TG={tg_in*100:.0f}%")
        else:
            st.warning(f"DCF: {dcf_res['error']}")

    # Graham Number
    with v_cols[1]:
        st.markdown("**📐 Graham Number**")
        graham = compute_graham_number(info)
        if graham:
            g_up = (graham / price - 1) * 100
            g_col= "#4caf50" if g_up > 0 else "#f44336"
            st.markdown(f"<h3 style='color:{g_col};'>{cur}{graham:,.2f}</h3>", unsafe_allow_html=True)
            st.metric("Upside", f"{g_up:+.1f}%")
            st.caption("√(22.5 × EPS × Book Value)  — Benjamin Graham")
        else:
            st.warning("EPS or Book Value not available.")

    # P/E Mean Reversion
    with v_cols[2]:
        st.markdown("**📊 P/E Reversion**")
        sector_pe_map = {"Technology":35,"Financials":18,"Healthcare":28,"Industrials":24,
                          "Energy":12,"Consumer Cyclical":22,"Consumer Defensive":30,"Utilities":20}
        median_pe = sector_pe_map.get(sector, 22)
        eps_val   = info.get("trailingEps")
        pe_fair   = round(eps_val * median_pe, 2) if eps_val and eps_val > 0 else None
        if pe_fair:
            pe_up = (pe_fair / price - 1) * 100
            pe_col= "#4caf50" if pe_up > 0 else "#f44336"
            st.markdown(f"<h3 style='color:{pe_col};'>{cur}{pe_fair:,.2f}</h3>", unsafe_allow_html=True)
            st.metric("Upside", f"{pe_up:+.1f}%")
            st.caption(f"Sector median P/E = {median_pe}x")
        else:
            st.warning("EPS not available.")

    # Summary
    st.divider()
    st.subheader("Valuation Summary")
    sum_data = {
        "Current Price": f"{cur}{price:,.2f}",
        "DCF Intrinsic":   f"{cur}{dcf_res['intrinsic']:,.2f}  ({dcf_res['upside']:+.1f}%)" if not dcf_res.get("error") else "N/A",
        "Graham Number":   f"{cur}{graham:,.2f}  ({g_up:+.1f}%)" if graham else "N/A",  # type: ignore
        "P/E Reversion":   f"{cur}{pe_fair:,.2f}  ({pe_up:+.1f}%)" if pe_fair else "N/A",  # type: ignore
        "Sector P/E avg":  f"{median_pe}x",
    }
    st.dataframe(pd.DataFrame.from_dict(sum_data, orient="index", columns=["Value"]),
                  use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 8 — TECHNICAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("Technical Analysis")
    d_t = df_ta.tail(252).copy()

    # Signal summary
    last = d_t.iloc[-1]
    rsi_s  = "Overbought" if last.get("RSI",50) > 70 else ("Oversold" if last.get("RSI",50) < 30 else "Neutral")
    macd_s = "Bullish" if (last.get("MACD",0) or 0) > (last.get("MACD_Sig",0) or 0) else "Bearish"
    adx_s  = "Trending" if (last.get("ADX",0) or 0) > 25 else "Ranging"

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("RSI (14)", f"{last.get('RSI',0):.1f}", rsi_s)
    sc2.metric("MACD",     f"{last.get('MACD',0):.3f}", macd_s)
    sc3.metric("ADX",      f"{last.get('ADX',0):.1f}", adx_s)
    sc4.metric("ATR",      f"{last.get('ATR',0):.2f}")

    # Main chart: Price + BB + RSI + MACD
    fig_t = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           row_heights=[0.45,0.18,0.18,0.19], vertical_spacing=0.03,
                           subplot_titles=["Price + BB","RSI (14)","MACD","Stochastic"])

    fig_t.add_trace(go.Candlestick(
        x=d_t.index, open=d_t["Open"], high=d_t["High"],
        low=d_t["Low"], close=d_t["Close"],
        increasing_line_color="#4caf50", decreasing_line_color="#f44336", name="Price"), row=1, col=1)
    for band, col_b, fill in [("BB_Upper","rgba(88,166,255,0.6)",None),
                                ("BB_Middle","rgba(249,168,37,0.8)",None),
                                ("BB_Lower","rgba(88,166,255,0.6)","tonexty")]:
        if band in d_t:
            fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t[band], name=band,
                                        line=dict(color=col_b, width=1),
                                        fill=fill, fillcolor="rgba(88,166,255,0.05)",
                                        showlegend=False), row=1, col=1)

    if "RSI" in d_t:
        fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t["RSI"], name="RSI",
                                    line=dict(color="#ce93d8", width=1.5)), row=2, col=1)
        for lv_r, c_r in [(70,"#f44336"),(30,"#4caf50")]:
            fig_t.add_hline(y=lv_r, line_color=c_r, line_dash="dot", opacity=0.5,
                             row=2, col=1, exclude_empty_subplots=False)

    if "MACD" in d_t:
        mc_cols = ["#4caf50" if v >= 0 else "#f44336" for v in d_t["MACD_Hist"].fillna(0)]
        fig_t.add_trace(go.Bar(x=d_t.index, y=d_t["MACD_Hist"], marker_color=mc_cols, showlegend=False), row=3, col=1)
        fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t["MACD"],     name="MACD",   line=dict(color="#58a6ff",width=1.2)), row=3, col=1)
        fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t["MACD_Sig"], name="Signal", line=dict(color="#f9a825",width=1.2)), row=3, col=1)

    if "Stoch_K" in d_t:
        fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t["Stoch_K"], name="%K", line=dict(color="#58a6ff",width=1.2)), row=4, col=1)
        fig_t.add_trace(go.Scatter(x=d_t.index, y=d_t["Stoch_D"], name="%D", line=dict(color="#f9a825",width=1.2)), row=4, col=1)
        for lv_s in [80,20]:
            fig_t.add_hline(y=lv_s, line_color="#8b949e", line_dash="dot", opacity=0.4,
                             row=4, col=1, exclude_empty_subplots=False)

    fig_t.update_layout(template="plotly_dark", height=780, paper_bgcolor="#0d1117",
                         xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_t, use_container_width=True)

    # OBV + Volume
    if "OBV" in d_t:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5,0.5],
                               subplot_titles=["OBV","CMF"])
        fig_v.add_trace(go.Scatter(x=d_t.index, y=d_t["OBV"],
                                    line=dict(color="#58a6ff",width=1.5), name="OBV"), row=1, col=1)
        if "CMF" in d_t:
            cmf_c = ["#4caf50" if v >= 0 else "#f44336" for v in d_t["CMF"].fillna(0)]
            fig_v.add_trace(go.Bar(x=d_t.index, y=d_t["CMF"], marker_color=cmf_c, name="CMF"), row=2, col=1)
            fig_v.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.3,
                             row=2, col=1, exclude_empty_subplots=False)
        fig_v.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0d1117")
        st.plotly_chart(fig_v, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 9 — SCREENER
# ════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("Multi-Stock Screener")
    st.caption("Like screener.in — filter by any combination of ratios across ~60 NSE stocks")

    with st.expander("⚙️ Configure Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        pe_max  = fc1.number_input("P/E ≤", value=40.0, min_value=0.0)
        pb_max  = fc1.number_input("P/B ≤", value=8.0,  min_value=0.0)
        roe_min = fc2.number_input("ROE % ≥", value=10.0, min_value=0.0)
        nm_min  = fc2.number_input("Net Margin % ≥", value=5.0, min_value=0.0)
        de_max  = fc3.number_input("D/E ≤", value=3.0, min_value=0.0)
        mc_min  = fc3.number_input("Mkt Cap (Cr) ≥", value=500.0, min_value=0.0)
        rg_min  = fc1.number_input("Rev Growth % ≥", value=-50.0)
        dy_min  = fc2.number_input("Div Yield % ≥", value=0.0)
        run_screen = st.button("🔍 Run Screener")

    if run_screen:
        with st.spinner(f"Fetching data for {len(DEFAULT_UNIVERSE)} stocks…"):
            df_universe = fetch_screener_data(tuple(DEFAULT_UNIVERSE))
            filters = {
                "P/E":          ("<=", pe_max),
                "P/B":          ("<=", pb_max),
                "ROE %":        (">=", roe_min),
                "Net Margin %": (">=", nm_min),
                "D/E":          ("<=", de_max),
                "Mkt Cap (Cr)": (">=", mc_min),
                "Rev Growth %": (">=", rg_min),
                "Div Yield %":  (">=", dy_min),
            }
            df_result = apply_filters(df_universe, filters)
            st.session_state["screen_result"] = df_result

    if "screen_result" in st.session_state:
        df_r = st.session_state["screen_result"]
        st.success(f"✅  **{len(df_r)} stocks** passed all filters")
        if not df_r.empty:
            sort_col = st.selectbox("Sort by", ["ROE %","P/E","Mkt Cap (Cr)","Net Margin %"], index=0)
            df_r = df_r.sort_values(sort_col, ascending=False).reset_index(drop=True)
            st.dataframe(
                df_r.style
                .background_gradient(subset=["ROE %","Net Margin %"], cmap="Greens")
                .background_gradient(subset=["P/E","D/E"], cmap="RdYlGn_r")
                .format(na_rep="—"),
                use_container_width=True, height=500
            )
            st.download_button("⬇ Download CSV", df_r.to_csv(index=False),
                                file_name=f"screener_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 10 — CALCULATOR
# ════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("Investment Calculator")
    st.caption("Like groww.in — SIP, Lump-sum, Returns calculator")

    # Historical CAGR
    if len(hist) >= 252:
        n_y   = (hist.index[-1] - hist.index[0]).days / 365.25
        h_cagr= max(0, (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0])) ** (1/n_y) - 1)
    else:
        h_cagr = 0.12

    cc1, cc2 = st.columns([1, 2])
    with cc1:
        sip_amt   = st.number_input("Monthly SIP (₹)", value=10000, step=1000, min_value=500)
        lump_amt  = st.number_input("Lump-sum (₹)", value=100000, step=10000)
        inv_years = st.slider("Investment Horizon (years)", 1, 30, 10)
        cagr_in   = st.slider(f"Expected CAGR %  (historical: {h_cagr*100:.1f}%)", 4, 40, max(8, int(h_cagr*100)))
        run_calc  = st.button("Calculate Returns")

    with cc2:
        if run_calc or "calc_done" in st.session_state:
            if run_calc:
                st.session_state["calc_done"] = True
                st.session_state["calc_params"] = (sip_amt, lump_amt, inv_years, cagr_in)
            sip_a, lump_a, inv_y, cagr_p = st.session_state["calc_params"]
            r = cagr_p / 100
            mr = r / 12
            nm = inv_y * 12

            sip_fv  = sip_a  * (((1+mr)**nm - 1)/mr) * (1+mr)
            ls_fv   = lump_a * (1+r) ** inv_y
            sip_inv = sip_a  * nm

            res_cols = st.columns(2)
            res_cols[0].metric("SIP Final Value",  f"₹{sip_fv:,.0f}", f"Invested ₹{sip_inv:,.0f}")
            res_cols[1].metric("Lump-sum Final",   f"₹{ls_fv:,.0f}",  f"Invested ₹{lump_a:,.0f}")

            months  = list(range(1, nm+1))
            sip_c   = [sip_a * (((1+mr)**m - 1)/mr) * (1+mr) for m in months]
            ls_c    = [lump_a * (1+r)**(m/12) for m in months]
            invested= [sip_a * m for m in months]

            fig_calc = go.Figure()
            fig_calc.add_trace(go.Scatter(x=months, y=sip_c,    name=f"SIP ₹{sip_a:,}/mo",   line=dict(color="#4caf50", width=2)))
            fig_calc.add_trace(go.Scatter(x=months, y=ls_c,     name=f"Lump-sum ₹{lump_a:,}", line=dict(color="#58a6ff", width=2)))
            fig_calc.add_trace(go.Scatter(x=months, y=invested,  name="SIP Invested",          line=dict(color="#8b949e", width=1, dash="dash")))
            fig_calc.update_layout(template="plotly_dark", height=360, paper_bgcolor="#0d1117",
                                    xaxis_title="Months", yaxis_title="Value (₹)",
                                    title=f"Investment Growth @ {cagr_p}% CAGR over {inv_y} years")
            st.plotly_chart(fig_calc, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"SCREENER  ·  Data from Yahoo Finance  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
    "For informational purposes only — not financial advice"
)
