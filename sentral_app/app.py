"""
SENTRAL — Multi-Spectrum Stock Analysis
Streamlit app: mirrors all functionality of SENTRAL_Complete.ipynb
Run with: streamlit run app.py
"""
import os, warnings, time, io, zipfile, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv, set_key, find_dotenv

# ── Module imports ─────────────────────────────────────────────────────────────
from modules.data_loader   import download_stock_data, compute_features
from modules.fundamental   import (compute_fundamental_metrics, run_dcf,
                                   run_altman_z, run_piotroski, compute_fundamental_score)
from modules.technical     import (compute_risk_metrics, detect_candlestick_patterns,
                                   get_pattern_summary, compute_seasonality,
                                   compute_technical_score, get_risk_flags)
from modules.peers         import get_peer_data
from modules.news          import fetch_all_news
from modules.sentiment     import (run_vader, run_finbert, run_finbert_tone,
                                   run_distilroberta, run_roberta_large,
                                   run_stocktwits_roberta, run_groq, run_gemini,
                                   aggregate_sentiment, generate_thesis)
from modules.ml_forecast   import run_ml_forecast, run_monte_carlo
from modules.signals       import compute_signal
from modules.backtest      import run_backtest, compute_kelly
from modules.report        import generate_html_report, generate_pdf_report

# ─────────────────────────────────────────────────────────────────────────────
#  Bundle ZIP: save & restore full analysis sessions
# ─────────────────────────────────────────────────────────────────────────────
def _json_float(obj):
    """JSON serializer that handles numpy/pandas scalar types."""
    if isinstance(obj, (np.integer,)):             return int(obj)
    if isinstance(obj, (np.floating, np.float_)):  return float(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, (np.bool_,)):               return bool(obj)
    return str(obj)


def _build_bundle_zip() -> bytes:
    """Package all current session analysis results into a downloadable ZIP."""
    ss = st.session_state
    df           = ss["df"]
    df_news      = ss["df_news"]
    df_sentiment = ss["df_sentiment"]
    df_peers     = ss["df_peers"]
    bt           = ss["bt_results"]
    ml           = ss["ml"]
    raw          = ss["raw"]
    mc           = ss["mc"]
    risk         = ss["risk"]

    session_json = {
        "ticker":           ss.get("_ticker", ""),
        "company_name":     raw["company_name"],
        "sector":           raw["sector"],
        "industry":         raw["industry"],
        "exchange":         raw["exchange"],
        "currency":         raw.get("currency", ""),
        "currency_sym":     raw["currency_sym"],
        "period":           ss.get("_period", "2y"),
        "forecast_days":    int(ss.get("_forecast_days", 30)),
        "mc_sims":          int(mc["n_sims"]),
        "signal":           ss["signal_data"],
        "altman":           ss["altman"],
        "piotroski_score":  ss["piotroski"]["score"],
        "piotroski_signal": ss["piotroski"]["signal"],
        "piotroski_criteria": {k: bool(v) for k, v in ss["piotroski"].get("criteria", {}).items()},
        "dcf":              {k: v for k, v in ss["dcf"].items()},
        "risk_scalars":     {k: v for k, v in risk.items()
                             if not isinstance(v, (pd.Series, pd.DataFrame))},
        "kelly":            ss["kelly"],
        "metrics":          ss["metrics"],
        "thesis":           ss["thesis"],
        "ensemble_score":   float(ss["ens_score"]),
        "ensemble_label":   ss["ens_label"],
        "peer_method":      ss.get("peer_method", "sector-database"),
        "auto_peers":       ss["auto_peers"],
        "mc_summary": {
            "p_profit": mc["p_profit"], "p_gain5": mc["p_gain5"],
            "p_loss5":  mc["p_loss5"],  "pctiles":  mc["pctiles"],
            "S0": mc["S0"], "mu_ann": mc["mu_ann"], "sig_ann": mc["sig_ann"],
            "n_sims": mc["n_sims"], "days": mc["days"],
        },
        "ml_summary": {
            "lstm_30d":        ml["lstm_30d"],
            "trans_30d":       ml["trans_30d"],
            "ensemble_30d":    ml["ensemble_30d"],
            "forecast_return": ml["forecast_return"],
            "current_price":   ml["current_price"],
        } if ml else None,
        "fund_score":  float(ss.get("fund_score", 0)),
        "tech_score":  float(ss.get("tech_score", 0)),
        "risk_flags":  ss.get("risk_flags", []),
        "bt_best_name": bt["best_name"],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("price_indicators.csv", df.to_csv())
        if not df_news.empty:
            zf.writestr("news_corpus.csv", df_news.to_csv(index=False))
        if not df_sentiment.empty:
            zf.writestr("sentiment_analysis.csv", df_sentiment.to_csv())
        if not df_peers.empty:
            zf.writestr("peers.csv", df_peers.to_csv())
        zf.writestr("backtest_results.csv", bt["df_results"].to_csv())
        cumuls_df = pd.DataFrame(bt["cumuls"], index=bt["bt_index"])
        cumuls_df["★ Buy & Hold"] = bt["bh_cumul"].values
        zf.writestr("backtest_cumuls.csv", cumuls_df.to_csv())
        if ml:
            ml_df = pd.DataFrame({
                "date":                 [str(d) for d in ml["future_dates"]],
                "lstm_forecast":        list(ml["lstm_future"]),
                "transformer_forecast": list(ml["trans_future"]),
                "ensemble_forecast":    list(ml["ensemble_future"]),
            })
            zf.writestr("ml_forecast.csv", ml_df.to_csv(index=False))
            zf.writestr("ml_eval.csv",     ml["eval_df"].to_csv())
        zf.writestr("session_data.json",
                    json.dumps(session_json, default=_json_float, indent=2))
    buf.seek(0)
    return buf.read()


def _restore_from_bundle(uploaded_file) -> bool:
    """
    Extract a bundle ZIP, rebuild session_state, return True on success.
    Expensive computations (news / sentiment / ML / backtest) are loaded from
    the ZIP.  Cheap ones (risk metrics, patterns, seasonality, Monte Carlo) are
    re-run from the saved price_indicators.csv (~1-2 s total).
    yfinance raw data is fetched via download_stock_data (24 h cache, fast).
    """
    try:
        raw_bytes = uploaded_file.read()
        buf = io.BytesIO(raw_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()

            def _rc(fname, **kw):
                if fname not in names:
                    return pd.DataFrame()
                return pd.read_csv(io.BytesIO(zf.read(fname)), **kw)

            # ── session_data.json ──────────────────────────────────────────
            if "session_data.json" not in names:
                st.error("Invalid bundle: missing session_data.json")
                return False
            sd = json.loads(zf.read("session_data.json").decode("utf-8"))

            ticker_b        = sd["ticker"]
            period_b        = sd["period"]
            forecast_days_b = int(sd.get("forecast_days", 30))
            mc_sims_b       = int(sd.get("mc_sims", 1000))

            # ── price_indicators.csv ───────────────────────────────────────
            df = _rc("price_indicators.csv", index_col=0, parse_dates=True)
            if df.empty:
                st.error("Invalid bundle: missing price_indicators.csv")
                return False
            df.index = pd.to_datetime(df.index)
            close = df["Close"]

            # ── Re-run cheap technical computations (no API calls) ─────────
            risk        = compute_risk_metrics(df)
            patterns    = detect_candlestick_patterns(df)
            pat_summ    = get_pattern_summary(patterns, df)
            seasonality = compute_seasonality(df)
            tech_score  = float(sd.get("tech_score") or compute_technical_score(df))
            risk_flags  = sd.get("risk_flags") or get_risk_flags(df)

            # ── Monte Carlo (pure numpy, ~1 s) ─────────────────────────────
            mc = run_monte_carlo(close, forecast_days=forecast_days_b,
                                 n_sims=mc_sims_b)

            # ── yfinance raw data (24 h @st.cache_data) ───────────────────
            with st.spinner(f"Refreshing {ticker_b} market data (cached)…"):
                raw = download_stock_data(ticker_b, period_b)

            # ── Load saved CSVs ────────────────────────────────────────────
            df_news      = _rc("news_corpus.csv")
            df_sentiment = _rc("sentiment_analysis.csv", index_col=0)
            df_peers     = _rc("peers.csv", index_col=0)

            # ── Backtest results ───────────────────────────────────────────
            df_bt  = _rc("backtest_results.csv", index_col=0)
            cum_df = _rc("backtest_cumuls.csv",  index_col=0, parse_dates=True)
            if not cum_df.empty:
                cum_df.index = pd.to_datetime(cum_df.index)
            bt_index = cum_df.index if not cum_df.empty else pd.DatetimeIndex([])
            bh_col   = "★ Buy & Hold"
            if not cum_df.empty and bh_col in cum_df.columns:
                bh_cumul = cum_df.pop(bh_col)
            else:
                bh_cumul = pd.Series(dtype=float)
            cumuls = {c: cum_df[c] for c in cum_df.columns} if not cum_df.empty else {}
            bt_results = {
                "df_results": df_bt,
                "cumuls":     cumuls,
                "bh_cumul":   bh_cumul,
                "bt_index":   bt_index,
                "best_name":  sd.get("bt_best_name", ""),
                "best_rets":  pd.Series(dtype=float),
            }

            # ── ML forecast ───────────────────────────────────────────────
            ml = None
            if "ml_forecast.csv" in names:
                ml_df   = _rc("ml_forecast.csv")
                ml_eval = _rc("ml_eval.csv", index_col=0)
                mls     = sd.get("ml_summary") or {}
                if mls and not ml_df.empty:
                    fallback_eval = pd.DataFrame(
                        {"RMSE": [0, 0], "MAPE %": ["N/A", "N/A"]},
                        index=pd.Index(["LSTM", "Transformer"], name="Model"))
                    ml = {
                        "lstm_future":     ml_df["lstm_forecast"].to_numpy(),
                        "trans_future":    ml_df["transformer_forecast"].to_numpy(),
                        "ensemble_future": ml_df["ensemble_forecast"].to_numpy(),
                        "future_dates":    pd.to_datetime(ml_df["date"]),
                        "lstm_30d":        float(mls.get("lstm_30d", 0)),
                        "trans_30d":       float(mls.get("trans_30d", 0)),
                        "ensemble_30d":    float(mls.get("ensemble_30d", 0)),
                        "forecast_return": float(mls.get("forecast_return", 0.0)),
                        "current_price":   float(mls.get("current_price",
                                                          float(close.iloc[-1]))),
                        "eval_df":         ml_eval if not ml_eval.empty else fallback_eval,
                        "lstm_preds":  np.array([]), "trans_preds": np.array([]),
                        "actuals":     np.array([]), "test_dates":  pd.DatetimeIndex([]),
                        "lstm_model": None, "trans_model": None,
                        "scaler_close": None, "scaler_feat": None,
                    }

            # ── Reconstruct piotroski dict ─────────────────────────────────
            piotroski = {
                "score":    sd.get("piotroski_score", 0),
                "signal":   sd.get("piotroski_signal", ""),
                "criteria": sd.get("piotroski_criteria", {}),
            }

            # ── Rebuild all_results for reports ───────────────────────────
            all_results = {
                "ticker":       ticker_b,
                "company_name": sd["company_name"],
                "currency_sym": sd["currency_sym"],
                "info":         raw["info"],
                "signal":       sd["signal"],
                "dcf":          sd["dcf"],
                "altman":       sd["altman"],
                "piotroski":    piotroski,
                "risk":         risk,
                "sentiment_df": df_sentiment,
                "thesis":       sd.get("thesis", ""),
                "news_items":   df_news.to_dict("records") if not df_news.empty else [],
                "backtest":     bt_results,
            }
            with st.spinner("Regenerating reports…"):
                html_report = generate_html_report(all_results)
                pdf_report  = generate_pdf_report(all_results)

        # ── Populate session_state ─────────────────────────────────────────
        st.session_state.clear()
        ss = st.session_state
        ss["raw"]            = raw
        ss["df"]             = df
        ss["metrics"]        = sd.get("metrics", {})
        ss["altman"]         = sd["altman"]
        ss["piotroski"]      = piotroski
        ss["dcf"]            = sd["dcf"]
        ss["fund_score"]     = float(sd.get("fund_score", 0))
        ss["risk"]           = risk
        ss["patterns"]       = patterns
        ss["pat_summ"]       = pat_summ
        ss["seasonality"]    = seasonality
        ss["tech_score"]     = tech_score
        ss["risk_flags"]     = risk_flags
        ss["auto_peers"]     = sd.get("auto_peers", [])
        ss["df_peers"]       = df_peers
        ss["peer_method"]    = sd.get("peer_method", "sector-database")
        ss["news_items"]     = df_news.to_dict("records") if not df_news.empty else []
        ss["df_news"]        = df_news
        ss["sent_results"]   = {}
        ss["df_sentiment"]   = df_sentiment
        ss["ens_score"]      = float(sd.get("ensemble_score", 0.0))
        ss["ens_label"]      = sd.get("ensemble_label", "Neutral")
        ss["thesis"]         = sd.get("thesis", "No thesis available.")
        ss["ml"]             = ml
        ss["forecast_return"] = float((sd.get("ml_summary") or {}).get("forecast_return", 0.0))
        ss["mc"]             = mc
        ss["signal_data"]    = sd["signal"]
        ss["bt_results"]     = bt_results
        ss["kelly"]          = sd.get("kelly", {})
        ss["html_report"]    = html_report
        ss["pdf_report"]     = pdf_report
        ss["all_results"]    = all_results
        ss["_ticker"]        = ticker_b
        ss["_period"]        = period_b
        ss["_forecast_days"] = forecast_days_b
        ss["loaded_ticker"]  = ticker_b
        ss["completed"]      = True
        return True

    except Exception as exc:
        st.error(f"Failed to load bundle: {exc}")
        st.exception(exc)
        return False


st.set_page_config(
    page_title="SENTRAL Stock Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme override ────────────────────────────────────────────────────────
st.markdown("""<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }
  .stButton>button { background-color: #238636; color: white; border: none;
                     border-radius: 6px; font-weight: bold; }
  .stButton>button:hover { background-color: #2ea043; }
  .metric-row { display:flex; gap:16px; flex-wrap:wrap; }
  .metric-box { background:#161b22; border:1px solid #30363d; border-radius:8px;
                padding:14px 18px; min-width:140px; text-align:center; }
  .metric-box .val { font-size:20px; font-weight:bold; color:#f0f6fc; }
  .metric-box .lbl { font-size:11px; color:#8b949e; margin-top:4px; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stock-market.png", width=60)
    st.title("SENTRAL")
    st.caption("Multi-Spectrum Stock Analysis")
    st.divider()

    # ── Stock & Config ─────────────────────────────────────────────────────────
    st.subheader("📈 Stock")
    ticker    = st.text_input("Ticker Symbol", value="HAL.NS",
                               help="NSE: HAL.NS | US: AAPL | BSE: RELIANCE.BO")
    period    = st.selectbox("Historical Period",
                              ["1y","2y","3y","5y","10y","max"], index=1)
    benchmark = st.text_input("Benchmark Index", value="^NSEI",
                               help="^NSEI (Nifty), ^GSPC (S&P 500)")

    st.subheader("🔭 Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 120, 30)
    seq_len       = st.slider("LSTM Look-back (trading days)", 20, 120, 60)
    mc_sims       = st.slider("Monte Carlo Simulations", 500, 5000, 1000, step=500)

    st.subheader("⚙️ ML Options")
    run_ml = st.toggle("Train LSTM + Transformer (~30-60s)", value=True)

    st.divider()

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.subheader("🔑 API Keys")
    load_dotenv()

    def _key(label, env_var, type="password"):
        return st.text_input(label, value=os.getenv(env_var,""),
                              type=type, key=env_var)

    alpha_key   = _key("Alpha Vantage",  "ALPHA_VANTAGE_API_KEY")
    finnhub_key = _key("Finnhub",        "FINNHUB_API_KEY")
    tavily_key  = _key("Tavily",         "TAVILY_API_KEY")
    newsapi_key = _key("NewsAPI",        "NEWSAPI_KEY")
    eodhd_key   = _key("EODHD",         "EODHD_API_KEY")
    marketaux_k = _key("Marketaux",      "MARKETAUX_API_KEY")
    apilayer_k  = _key("APILayer",       "APILAYER_API_KEY")
    hf_token    = _key("HuggingFace",    "HF_TOKEN")
    groq_key    = _key("Groq",           "GROQ_API_KEY")
    gemini_key  = _key("Gemini",         "GEMINI_API_KEY")

    save_env = st.checkbox("💾 Save keys to .env", value=False)
    if save_env and st.button("Save Now"):
        env_file = find_dotenv() or os.path.join(os.path.dirname(__file__), ".env")
        keys_map = {
            "ALPHA_VANTAGE_API_KEY": alpha_key, "FINNHUB_API_KEY": finnhub_key,
            "TAVILY_API_KEY": tavily_key,        "NEWSAPI_KEY": newsapi_key,
            "EODHD_API_KEY": eodhd_key,          "MARKETAUX_API_KEY": marketaux_k,
            "APILAYER_API_KEY": apilayer_k,      "HF_TOKEN": hf_token,
            "GROQ_API_KEY": groq_key,            "GEMINI_API_KEY": gemini_key,
        }
        for k, v in keys_map.items():
            if v: set_key(env_file, k, v)
        st.success("Keys saved to .env")

    st.divider()
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

    st.divider()
    st.subheader("📂 Load Previous Analysis")
    st.caption("Upload a SENTRAL bundle ZIP to instantly reload a completed analysis — "
               "skipping news scraping, sentiment models, ML training, and backtesting.")
    uploaded_bundle = st.file_uploader("Upload Bundle ZIP", type=["zip"],
                                        label_visibility="collapsed",
                                        key="bundle_uploader")
    load_btn = st.button("⚡ Load from Bundle", use_container_width=True,
                          disabled=(uploaded_bundle is None))


api_keys = {
    "ALPHA_VANTAGE": alpha_key, "FINNHUB": finnhub_key,
    "TAVILY": tavily_key,       "NEWSAPI": newsapi_key,
    "EODHD": eodhd_key,         "MARKETAUX": marketaux_k,
    "APILAYER": apilayer_k,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 SENTRAL — Multi-Spectrum Stock Analysis")
st.caption("Fundamental · Technical · Sentiment · ML Forecast · Backtesting")

if not run_btn and not (load_btn and uploaded_bundle) and not st.session_state.get("completed"):
    st.info("Enter a ticker and click **🚀 Run Full Analysis** to begin.  \n"
            "Or upload a previous SENTRAL bundle ZIP in the sidebar to reload instantly.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ANALYSIS (triggered by button)
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    st.session_state.clear()
    progress = st.progress(0, text="Initialising…")
    status   = st.status("Running SENTRAL analysis…", expanded=True)

    def upd(pct, msg):
        progress.progress(pct, text=msg)
        status.write(f"**{msg}**")

    try:
        # ── 1. Data Download ──────────────────────────────────────────────────
        upd(2, "⬇ Downloading market data…")
        raw = download_stock_data(ticker.strip().upper(), period)
        df  = compute_features(raw["data"])
        close, high, low = df["Close"], df["High"], df["Low"]
        vol              = df["Volume"]
        info             = raw["info"]
        company_name     = raw["company_name"]
        cur_sym          = raw["currency_sym"]
        st.session_state["raw"] = raw
        st.session_state["df"]  = df
        st.session_state["_ticker"]       = ticker.strip().upper()
        st.session_state["_period"]       = period
        st.session_state["_forecast_days"] = forecast_days

        upd(8, "📐 Computing fundamental metrics…")
        # ── 2. Fundamental ────────────────────────────────────────────────────
        metrics    = compute_fundamental_metrics(info, raw["fin"], raw["bal"], raw["cf"], cur_sym)
        altman     = run_altman_z(raw["bal"], raw["fin"], info)
        piotroski  = run_piotroski(raw["bal"], raw["fin"], raw["cf"])
        dcf        = run_dcf(raw["cf"], info, cur_sym)
        fund_score = compute_fundamental_score(info, piotroski["score"], altman["z_score"])
        st.session_state["metrics"]   = metrics
        st.session_state["altman"]    = altman
        st.session_state["piotroski"] = piotroski
        st.session_state["dcf"]       = dcf
        st.session_state["fund_score"] = fund_score

        upd(18, "📉 Computing technical indicators…")
        # ── 3. Technical ──────────────────────────────────────────────────────
        risk       = compute_risk_metrics(df)
        patterns   = detect_candlestick_patterns(df)
        pat_summ   = get_pattern_summary(patterns, df)
        seasonality = compute_seasonality(df)
        tech_score  = compute_technical_score(df)
        risk_flags  = get_risk_flags(df)
        st.session_state["risk"]        = risk
        st.session_state["patterns"]    = patterns
        st.session_state["pat_summ"]    = pat_summ
        st.session_state["seasonality"] = seasonality
        st.session_state["tech_score"]  = tech_score
        st.session_state["risk_flags"]  = risk_flags

        upd(25, "👥 Fetching peer comparison (LLM-powered)…")
        # ── 4. Peers ──────────────────────────────────────────────────────────
        auto_peers, df_peers, peer_method = get_peer_data(
            ticker.upper(), raw["sector"],
            company_name=company_name,
            exchange=raw["exchange"],
            groq_key=groq_key,
            gemini_key=gemini_key,
        )
        st.session_state["auto_peers"]  = auto_peers
        st.session_state["df_peers"]    = df_peers
        st.session_state["peer_method"] = peer_method

        upd(32, "📰 Scraping news from 10+ sources…")
        # ── 5. News ───────────────────────────────────────────────────────────
        news_items = fetch_all_news(ticker.upper(), company_name, api_keys)
        df_news    = pd.DataFrame(news_items) if news_items else pd.DataFrame()
        st.session_state["news_items"] = news_items
        st.session_state["df_news"]    = df_news

        upd(42, "🤖 Running sentiment models (VADER)…")
        # ── 6. Sentiment ──────────────────────────────────────────────────────
        texts_hf  = [n["title"] for n in news_items[:10]]
        texts_llm = [n["title"] for n in news_items[:8]]

        sent_results = {}
        sent_results["NLTK_VADER"]  = run_vader([n["title"] for n in news_items[:15]])

        upd(46, "🤖 FinBERT…")
        sent_results["FinBERT"]          = run_finbert(texts_hf, hf_token)
        upd(50, "🤖 FinBERT-Tone (w/ fallback)…")
        sent_results["FinBERT_Tone"]     = run_finbert_tone(texts_hf, hf_token)
        upd(54, "🤖 DistilRoBERTa…")
        sent_results["DistilRoBERTa"]    = run_distilroberta(texts_hf, hf_token)
        upd(57, "🤖 RoBERTa-large…")
        sent_results["RoBERTa_large"]    = run_roberta_large(texts_hf, hf_token)
        upd(60, "🤖 StockTwits RoBERTa…")
        sent_results["StockTwits_RoBERTa"] = run_stocktwits_roberta(texts_hf, hf_token)

        upd(63, "🤖 Groq LLMs…")
        groq1, groq1_model = run_groq(texts_llm, groq_key, model_slot=0)
        sent_results["Llama3_Groq"]  = groq1
        groq2, groq2_model = run_groq(texts_llm, groq_key, model_slot=1)
        sent_results["Gemma2_Groq"]  = groq2

        upd(68, "🤖 Gemini…")
        gem1, gem1_model = run_gemini(texts_llm, gemini_key)
        sent_results["Gemini_Flash"] = gem1
        gem2, gem2_model = run_gemini(texts_llm, gemini_key, skip_model=gem1_model)
        sent_results["Gemini_Pro"]   = gem2

        df_sentiment, ens_score, ens_label = aggregate_sentiment(sent_results)
        st.session_state["sent_results"]  = sent_results
        st.session_state["df_sentiment"]  = df_sentiment
        st.session_state["ens_score"]     = ens_score
        st.session_state["ens_label"]     = ens_label

        upd(72, "✍️ Generating investment thesis…")
        top_headlines = [n["title"] for n in news_items[:15]]
        thesis        = generate_thesis(top_headlines, df_sentiment,
                                         ticker, company_name, ens_score, gemini_key)
        st.session_state["thesis"] = thesis

        # ── 7. ML Forecast ────────────────────────────────────────────────────
        if run_ml:
            upd(75, "🧠 Training LSTM…")
            def ml_progress(event, ep, tot):
                if event == "lstm_start":  upd(75, "🧠 Training LSTM…")
                elif event == "trans_start": upd(84, "🧠 Training Transformer…")

            ml     = run_ml_forecast(df, forecast_days=forecast_days,
                                      seq_len=seq_len, epochs=50,
                                      batch_size=32, progress_cb=ml_progress)
            forecast_return = ml["forecast_return"]
        else:
            ml = None
            forecast_return = 0.0

        st.session_state["ml"]             = ml
        st.session_state["forecast_return"] = forecast_return

        upd(88, "🎲 Monte Carlo simulation…")
        mc = run_monte_carlo(close, forecast_days=forecast_days, n_sims=mc_sims)
        st.session_state["mc"] = mc

        upd(91, "📡 Computing final signal…")
        # ── 8. Signal ─────────────────────────────────────────────────────────
        signal_data = compute_signal(fund_score, tech_score, ens_score, forecast_return)
        st.session_state["signal_data"] = signal_data

        upd(94, "📊 Running 20-strategy backtest…")
        # ── 9. Backtest ───────────────────────────────────────────────────────
        bt_results = run_backtest(df)
        atr_val    = float(df["ATR"].dropna().iloc[-1]) if "ATR" in df.columns else 100.0
        kelly      = compute_kelly(bt_results["best_rets"], float(close.iloc[-1]),
                                    atr_val)
        st.session_state["bt_results"] = bt_results
        st.session_state["kelly"]      = kelly

        upd(98, "📝 Generating reports…")
        # ── 10. Reports ───────────────────────────────────────────────────────
        all_results = {
            "ticker": ticker, "company_name": company_name,
            "currency_sym": cur_sym, "info": info,
            "signal": signal_data, "dcf": dcf,
            "altman": altman, "piotroski": piotroski,
            "risk": risk, "sentiment_df": df_sentiment,
            "thesis": thesis, "news_items": news_items,
            "backtest": bt_results,
        }
        html_report = generate_html_report(all_results)
        pdf_report  = generate_pdf_report(all_results)
        st.session_state["html_report"] = html_report
        st.session_state["pdf_report"]  = pdf_report
        st.session_state["all_results"] = all_results
        st.session_state["completed"]   = True

        upd(100, "✅ Analysis complete!")
        status.update(label="✅ Analysis complete!", state="complete")
        time.sleep(0.5)
        progress.empty()

    except Exception as e:
        status.update(label=f"❌ Error: {e}", state="error")
        st.exception(e)
        st.stop()

# ── Handle bundle upload ───────────────────────────────────────────────────────
if load_btn and uploaded_bundle is not None and not run_btn:
    ok = _restore_from_bundle(uploaded_bundle)
    if ok:
        st.rerun()
    st.stop()

#  RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("completed"):
    st.stop()

# Load from session state
raw          = st.session_state["raw"]
df           = st.session_state["df"]
info         = raw["info"]
company_name = raw["company_name"]
cur_sym      = raw["currency_sym"]
metrics      = st.session_state["metrics"]
altman       = st.session_state["altman"]
piotroski    = st.session_state["piotroski"]
dcf          = st.session_state["dcf"]
risk         = st.session_state["risk"]
patterns     = st.session_state["patterns"]
pat_summ     = st.session_state["pat_summ"]
seasonality  = st.session_state["seasonality"]
signal_data  = st.session_state["signal_data"]
df_sentiment = st.session_state["df_sentiment"]
ens_score    = st.session_state["ens_score"]
ens_label    = st.session_state["ens_label"]
thesis       = st.session_state["thesis"]
news_items   = st.session_state["news_items"]
df_news      = st.session_state["df_news"]
ml           = st.session_state["ml"]
mc           = st.session_state["mc"]
bt_results   = st.session_state["bt_results"]
kelly        = st.session_state["kelly"]
html_report  = st.session_state["html_report"]
pdf_report   = st.session_state["pdf_report"]
auto_peers   = st.session_state["auto_peers"]
df_peers     = st.session_state["df_peers"]
peer_method  = st.session_state.get("peer_method", "sector-database")
risk_flags   = st.session_state["risk_flags"]

close = df["Close"]
# Use ticker from the loaded bundle when available (overrides sidebar widget)
ticker_disp = st.session_state.get("loaded_ticker", ticker.strip().upper())
_ts = pd.Timestamp.now().strftime('%Y%m%d')

# ── Header ─────────────────────────────────────────────────────────────────────
signal     = signal_data["signal"]
sig_score  = signal_data["final_score"]
sig_conf   = signal_data["confidence_pct"]
sig_color  = {"BUY 🟢":"#2e7d32","SELL 🔴":"#c62828"}.get(signal,"#f57f17")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.subheader(f"{company_name}  ({ticker_disp})")
    st.caption(f"{raw['sector']} · {raw['industry']} · {raw['exchange']}")
with col2:
    st.markdown(f"""<div style='background:{sig_color};border-radius:8px;padding:16px;text-align:center;'>
        <div style='font-size:28px;font-weight:bold;color:white;'>{signal}</div>
        <div style='color:rgba(255,255,255,0.8);font-size:13px;'>Score: {sig_score:+.4f} | Confidence: {sig_conf:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    price = info.get("currentPrice") or info.get("regularMarketPrice") or float(close.iloc[-1])
    st.metric("Current Price", f"{cur_sym}{price:,.2f}")
    st.metric("Data Points", f"{len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")

st.divider()

# ── Download Buttons ──────────────────────────────────────────────────────────
dc1, dc2, dc3, dc4 = st.columns(4)
with dc1:
    st.download_button("⬇ HTML Report", html_report,
                        file_name=f"SENTRAL_{ticker_disp}_{_ts}.html",
                        mime="text/html", use_container_width=True)
with dc2:
    if pdf_report:
        st.download_button("⬇ PDF Report", pdf_report,
                            file_name=f"SENTRAL_{ticker_disp}_{_ts}.pdf",
                            mime="application/pdf", use_container_width=True)
    else:
        st.info("PDF: install `reportlab`")
with dc3:
    csv_data = df.to_csv().encode()
    st.download_button("⬇ Price + Indicators CSV", csv_data,
                        file_name=f"{ticker_disp}_indicators.csv",
                        mime="text/csv", use_container_width=True)
with dc4:
    bundle_bytes = _build_bundle_zip()
    st.download_button(
        "📦 Download Full Bundle ZIP",
        bundle_bytes,
        file_name=f"SENTRAL_{ticker_disp}_{_ts}_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        help="ZIP with news, sentiment, ML forecasts, backtest curves + all data. "
             "Upload back via the sidebar to skip reprocessing.",
    )

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Signal", "💰 Fundamentals", "📉 Technical",
    "🗞️ News & Sentiment", "🧠 ML Forecast", "📈 Backtest",
    "👥 Peers", "📰 Thesis"
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIGNAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Signal Engine — Pillar Breakdown")

    pillars = signal_data["pillar_scores"]
    fig_sig = make_subplots(rows=1, cols=2,
                             specs=[[{"type":"indicator"},{"type":"bar"}]],
                             subplot_titles=["Composite Score","Pillar Breakdown"],
                             column_widths=[0.42, 0.58])
    fig_sig.add_trace(go.Indicator(
        mode="gauge+number+delta", value=round(sig_score*100,1),
        title={"text":f"<b>{signal}</b>","font":{"size":22}},
        delta={"reference":0},
        domain={"row":0,"column":0},
        gauge={"axis":{"range":[-100,100]},
               "bar":{"color":sig_color},
               "steps":[{"range":[-100,-20],"color":"#c62828"},
                        {"range":[-20,20],"color":"#f9a825"},
                        {"range":[20,100],"color":"#2e7d32"}],
               "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":sig_score*100}}
    ), row=1, col=1)

    pv   = [v*100 for v in pillars.values()]
    bcol = ["#4caf50" if v>=0 else "#f44336" for v in pv]
    fig_sig.add_trace(go.Bar(
        x=list(pillars.keys()), y=pv, marker_color=bcol,
        text=[f"{v:+.1f}" for v in pv], textposition="outside",
        textfont=dict(size=13,color="white"), showlegend=False,
    ), row=1, col=2)
    fig_sig.update_yaxes(range=[-130,130], title_text="Score (−100 to +100)", row=1, col=2)
    for _y, _c, _d in [(20,"#4caf50","dot"),(-20,"#f44336","dot"),(0,"white","solid")]:
        fig_sig.add_hline(y=_y, line_color=_c, line_dash=_d, line_width=1.5,
                          opacity=0.6, row=1, col=2, exclude_empty_subplots=False)
    fig_sig.update_layout(template="plotly_dark", height=420, showlegend=False,
                           paper_bgcolor="#0d1117")
    st.plotly_chart(fig_sig, use_container_width=True)

    if risk_flags:
        for _flag in risk_flags:
            st.warning(_flag)
    else:
        st.success("✅ No major risk flags detected.")

    st.divider()
    st.subheader("📊 Risk Metrics")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Sharpe Ratio",    f"{risk['sharpe']:.3f}")
    c2.metric("Max Drawdown",    f"{risk['max_drawdown']:.2f}%")
    c3.metric("VaR 95%",         f"{risk['var_95']:.2f}%")
    c4.metric("CVaR 95%",        f"{risk['cvar_95']:.2f}%")
    c5.metric("Ann. Return",     f"{risk['ann_return']:.2f}%")
    c6.metric("Ann. Volatility", f"{risk['ann_vol']:.2f}%")

    # Drawdown chart
    dd = risk["drawdown"]
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd*100, fill="tozeroy",
                                 fillcolor="rgba(244,67,54,0.3)",
                                 line=dict(color="#f44336",width=1), name="Drawdown"))
    fig_dd.update_layout(title="Rolling Drawdown", template="plotly_dark",
                          height=280, yaxis_title="Drawdown (%)",
                          paper_bgcolor="#0d1117", margin=dict(t=40,b=30))
    st.plotly_chart(fig_dd, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FUNDAMENTALS
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Fundamental Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ⚖️ Altman Z-Score")
        z = altman["z_score"]
        zone_color = {"SAFE":"#4caf50","GREY":"#f9a825","DISTRESS":"#f44336"}.get(altman["zone"],"#aaa")
        st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center;'>
          <div style='font-size:36px;font-weight:bold;color:{zone_color};'>{z}</div>
          <div style='color:{zone_color};font-size:16px;'>{altman['zone']} ZONE</div>
          <div style='color:#8b949e;font-size:11px;margin-top:4px;'>&gt;2.99=Safe · 1.81–2.99=Grey · &lt;1.81=Distress</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 📋 Piotroski F-Score")
        fscore = piotroski["score"]
        f_color = "#4caf50" if fscore >= 7 else ("#f9a825" if fscore >= 4 else "#f44336")
        st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center;'>
          <div style='font-size:36px;font-weight:bold;color:{f_color};'>{fscore}/9</div>
          <div style='color:{f_color};font-size:16px;'>{piotroski['signal']}</div>
          <div style='color:#8b949e;font-size:11px;margin-top:4px;'>7-9=Strong · 4-6=Moderate · 0-3=Weak</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("#### 💹 DCF Valuation")
        if dcf.get("error"):
            st.warning(f"DCF: {dcf['error']}")
        elif dcf.get("intrinsic"):
            intr = dcf["intrinsic"]
            usd  = dcf["upside_pct"]
            dcf_color = "#4caf50" if usd and usd > 0 else "#f44336"
            st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center;'>
              <div style='font-size:22px;font-weight:bold;color:{dcf_color};'>{cur_sym}{intr:,.2f}</div>
              <div style='color:{dcf_color};font-size:15px;'>{dcf['signal']} ({usd:+.1f}%)</div>
              <div style='color:#8b949e;font-size:11px;margin-top:4px;'>FCF: {cur_sym}{dcf.get('fcf',0):,.1f} Cr</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 Key Metrics (38 indicators)")
    # Display metrics in a table
    def _fmt_metric(x):
        if x is None:
            return "N/A"
        try:
            v = float(x)
        except (TypeError, ValueError):
            return str(x) if x else "N/A"
        if v == 0:
            return "0"
        # Treat small decimals (<2) that look like rates as percentages
        pct_keys = ["margin", "yield", "growth", "ratio", "return", "roe", "roa", "roic"]
        return f"{v:,.4f}"
    m_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    m_df["Value"] = m_df["Value"].apply(_fmt_metric)
    st.dataframe(m_df, use_container_width=True, height=600)

    # Piotroski detail
    with st.expander("📋 Piotroski Criteria Detail"):
        p_criteria = piotroski.get("criteria", {})
        for k, v in p_criteria.items():
            icon = "✅" if v else "❌"
            st.write(f"{icon} {k.replace('_',' ')}")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TECHNICAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Technical Analysis")

    # Candlestick chart
    st.markdown("#### 🕯️ Price Chart (last 1 year)")
    hist_df = df.tail(252)
    fig_cs = go.Figure()
    fig_cs.add_trace(go.Candlestick(
        x=hist_df.index, open=hist_df["Open"], high=hist_df["High"],
        low=hist_df["Low"], close=hist_df["Close"],
        increasing_line_color="#4caf50", decreasing_line_color="#f44336",
        name="OHLC"))
    for col, color, dash in [("SMA_50","#ff9800","solid"),("SMA_200","#58a6ff","dash"),("EMA_20","#ab47bc","dot")]:
        if col in hist_df.columns:
            fig_cs.add_trace(go.Scatter(x=hist_df.index, y=hist_df[col],
                                         line=dict(color=color,width=1.2,dash=dash),
                                         name=col, opacity=0.8))
    fig_cs.update_layout(template="plotly_dark", height=450, paper_bgcolor="#0d1117",
                          xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_cs, use_container_width=True)

    # Indicator snapshot
    latest = df.iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("RSI (14)", f"{latest.get('RSI',float('nan')):.1f}")
    c2.metric("MACD",     f"{latest.get('MACD',0):.2f}")
    c3.metric("ATR (14)", f"{cur_sym}{latest.get('ATR',0):.2f}")
    c4.metric("ADX (14)", f"{latest.get('ADX',0):.1f}")

    st.divider()
    # Candlestick patterns
    st.markdown("#### 🕯️ Candlestick Patterns Detected")
    pat_data = {k: v for k, v in pat_summ.items() if not k.startswith("_")}
    pat_df   = pd.DataFrame(pat_data).T
    st.dataframe(pat_df, use_container_width=True)
    most_recent = pat_summ.get("_most_recent",{})
    if most_recent.get("pattern"):
        st.info(f"Most recent: **{most_recent['pattern'].replace('_',' ')}** on {most_recent['date']}")

    st.divider()
    # Seasonality
    st.markdown("#### 📅 Seasonality")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption("Monthly Returns")
        mo_data = seasonality["monthly"]
        mo_df   = pd.DataFrame(mo_data).T.reset_index().rename(columns={"index":"Month"})
        fig_mo  = px.bar(mo_df, x="Month", y="mean", color="mean",
                          color_continuous_scale=["#f44336","#f9a825","#4caf50"],
                          labels={"mean":"Avg Return (%)"},
                          title=f"Best: {seasonality['best_month']} | Worst: {seasonality['worst_month']}")
        fig_mo.update_layout(template="plotly_dark", height=280, paper_bgcolor="#0d1117",
                              showlegend=False, margin=dict(t=40,b=30))
        st.plotly_chart(fig_mo, use_container_width=True)
    with sc2:
        st.caption("Day-of-Week Returns")
        dow_data = seasonality["daily"]
        dow_df   = pd.DataFrame(dow_data).T.reset_index().rename(columns={"index":"Day"})
        fig_dow  = px.bar(dow_df, x="Day", y="mean", color="mean",
                          color_continuous_scale=["#f44336","#f9a825","#4caf50"],
                          labels={"mean":"Avg Return (%)"},
                          title=f"Best: {seasonality['best_day']} | Worst: {seasonality['worst_day']}")
        fig_dow.update_layout(template="plotly_dark", height=280, paper_bgcolor="#0d1117",
                               showlegend=False, margin=dict(t=40,b=30))
        st.plotly_chart(fig_dow, use_container_width=True)

    st.divider()
    # Correlation heatmap
    with st.expander("📊 Indicator Correlation Heatmap"):
        corr_cols = ["Close","RSI","MACD","ADX","BB_Pct","ATR","OBV","CMF","MFI","ROC"]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_df   = df[corr_cols].dropna().corr()
        fig_hm    = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                               title="Indicator Correlation Matrix")
        fig_hm.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0d1117")
        st.plotly_chart(fig_hm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — NEWS & SENTIMENT
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("News & Sentiment Analysis")
    st.info(f"Scraped **{len(news_items)} relevant articles** from 12+ sources")

    if not df_sentiment.empty:
        # Ensemble score
        ens_color = "#4caf50" if ens_score > 0.1 else ("#f44336" if ens_score < -0.1 else "#f9a825")
        st.markdown(f"""<div style='text-align:center;padding:12px;'>
          <span style='font-size:22px;font-weight:bold;color:{ens_color};'>
          Ensemble Sentiment: {ens_label} ({ens_score:+.3f})</span>
        </div>""", unsafe_allow_html=True)

        # Sentiment summary table
        st.dataframe(df_sentiment.style.background_gradient(
            subset=["avg_score"], cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True)

        # Bar chart
        fig_sent = px.bar(df_sentiment.reset_index(), x="index", y="avg_score",
                           color="avg_score",
                           color_continuous_scale=["#f44336","#f9a825","#4caf50"],
                           labels={"index":"Model","avg_score":"Score"},
                           title="Average Sentiment Score per Model")
        fig_sent.add_hline(y=0, line_color="white", line_dash="solid", opacity=0.4)
        fig_sent.update_layout(template="plotly_dark", height=350,
                                paper_bgcolor="#0d1117", showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()
    st.subheader("📰 News Articles")
    if not df_news.empty:
        cols_show = [c for c in ["title","source","published"] if c in df_news.columns]
        st.dataframe(df_news[cols_show].head(50), use_container_width=True)
    else:
        st.warning("No news articles fetched.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ML FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("ML Price Forecasting")

    if ml is None:
        st.info("ML training was disabled. Enable it in the sidebar and re-run.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("LSTM 30d Forecast",        f"{cur_sym}{ml['lstm_30d']:,.2f}",
                   f"{(ml['lstm_30d']/ml['current_price']-1)*100:+.2f}%")
        c2.metric("Transformer 30d Forecast", f"{cur_sym}{ml['trans_30d']:,.2f}",
                   f"{(ml['trans_30d']/ml['current_price']-1)*100:+.2f}%")
        c3.metric("Ensemble 30d Forecast",    f"{cur_sym}{ml['ensemble_30d']:,.2f}",
                   f"{ml['forecast_return']*100:+.2f}%")

        # Forecast chart
        fig_fc = go.Figure()
        hist_tail = df["Close"].tail(90)
        fig_fc.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail,
                                     name="Historical", line=dict(color="#58a6ff",width=2)))
        fig_fc.add_trace(go.Scatter(x=ml["future_dates"], y=ml["lstm_future"],
                                     name="LSTM", line=dict(color="#ff9800",dash="dash")))
        fig_fc.add_trace(go.Scatter(x=ml["future_dates"], y=ml["trans_future"],
                                     name="Transformer", line=dict(color="#ab47bc",dash="dot")))
        fig_fc.add_trace(go.Scatter(x=ml["future_dates"], y=ml["ensemble_future"],
                                     name="Ensemble", line=dict(color="#4caf50",width=2.5)))
        fig_fc.update_layout(title=f"{forecast_days}-Day Price Forecast",
                              template="plotly_dark", height=420, paper_bgcolor="#0d1117")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.subheader("Model Evaluation")
        st.dataframe(ml["eval_df"], use_container_width=True)

    st.divider()
    st.subheader(f"🎲 Monte Carlo Simulation ({mc['n_sims']:,} paths)")
    mc_x   = np.arange(mc["days"] + 1)
    fig_mc = make_subplots(rows=1, cols=2,
                            subplot_titles=["Fan Chart", f"Terminal Distribution ({mc['days']}d)"],
                            column_widths=[0.65, 0.35])
    for i in np.random.choice(mc["n_sims"], min(150, mc["n_sims"]), replace=False):
        fig_mc.add_trace(go.Scatter(x=mc_x, y=mc["paths"][i], mode="lines",
                                     line=dict(color="steelblue",width=0.4),
                                     opacity=0.04, showlegend=False), row=1, col=1)
    pp = mc["pct_paths"]
    fig_mc.add_trace(go.Scatter(x=mc_x, y=pp[50], name="Median",
                                 line=dict(color="yellow",width=2)), row=1, col=1)
    fig_mc.add_trace(go.Scatter(x=np.concatenate([mc_x, mc_x[::-1]]),
                                 y=np.concatenate([pp[95], pp[5][::-1]]),
                                 fill="toself", fillcolor="rgba(70,130,180,0.12)",
                                 line=dict(color="rgba(255,255,255,0)"),
                                 name="5–95%ile", showlegend=True), row=1, col=1)
    if ml:
        ens_full = np.concatenate([[mc["S0"]], ml["ensemble_future"]])[:mc["days"]+1]
        fig_mc.add_trace(go.Scatter(x=mc_x, y=ens_full, name="DL Ensemble",
                                     line=dict(color="#ff9800",dash="dash",width=2)), row=1, col=1)
    fig_mc.add_trace(go.Histogram(y=mc["terminal"], nbinsy=60, marker_color="steelblue",
                                   opacity=0.7, showlegend=False), row=1, col=2)
    fig_mc.add_hline(y=mc["S0"], line_color="white", line_width=1.5,
                      row=1, col=2, exclude_empty_subplots=False)
    fig_mc.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0d1117")
    st.plotly_chart(fig_mc, use_container_width=True)

    mc_c1, mc_c2, mc_c3 = st.columns(3)
    mc_c1.metric("P(Profit in 30d)", f"{mc['p_profit']:.1%}")
    mc_c2.metric("P(Gain > 5%)",     f"{mc['p_gain5']:.1%}")
    mc_c3.metric("P(Loss > 5%)",     f"{mc['p_loss5']:.1%}")
    mc_c1.metric("MC Median",        f"{cur_sym}{mc['pctiles'][50]:,.2f}")
    mc_c2.metric("MC 5th pct",       f"{cur_sym}{mc['pctiles'][5]:,.2f}")
    mc_c3.metric("MC 95th pct",      f"{cur_sym}{mc['pctiles'][95]:,.2f}")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — BACKTEST
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader(f"20-Strategy Backtesting — {ticker_disp}")
    df_bt = bt_results["df_results"]
    best  = bt_results["best_name"]

    # Styled table
    def _style(row):
        if row.name == "★ Buy & Hold": return ["background:#1a3a5c;color:gold"]*len(row)
        if row.name == best:           return ["background:#0d3b26;color:#4ade80"]*len(row)
        return [""]*len(row)

    disp = df_bt.copy()
    for col in ["Total Return","Max Drawdown","Win Rate","Time in Mkt%"]:
        disp[col] = disp[col].apply(lambda x: f"{x:+.1%}")
    disp["Sharpe"] = disp["Sharpe"].apply(lambda x: f"{x:.3f}")
    st.dataframe(disp.style.apply(_style, axis=1), use_container_width=True)
    st.success(f"⭐ Best strategy: **{best}**")

    # Equity curves
    top5    = [n for n in df_bt.index if n != "★ Buy & Hold"][:5]
    pal     = ["#00bcd4","#ff9800","#e040fb","#66bb6a","#ef5350"]
    fig_eq  = go.Figure()
    for i, nm in enumerate(top5):
        fig_eq.add_trace(go.Scatter(x=bt_results["bt_index"],
                                     y=bt_results["cumuls"][nm],
                                     name=nm, line=dict(color=pal[i],width=1.8)))
    fig_eq.add_trace(go.Scatter(x=bt_results["bt_index"],
                                 y=bt_results["bh_cumul"],
                                 name="★ Buy & Hold",
                                 line=dict(color="white",dash="dash",width=2)))
    fig_eq.update_layout(title="Top 5 Strategies vs Buy & Hold",
                          template="plotly_dark", height=420, paper_bgcolor="#0d1117")
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()
    st.subheader("⚖️ Kelly Criterion Position Sizing")
    kc1,kc2,kc3,kc4 = st.columns(4)
    kc1.metric("Best Strategy Win Rate", f"{kelly['p_win']:.1%}")
    kc2.metric("Kelly Fraction",         f"{kelly['kelly_f']:.1%}")
    kc3.metric("Half-Kelly (advised)",   f"{kelly['half_kelly']:.1%}")
    kc4.metric("Suggested Allocation",   f"{cur_sym}{kelly['allocation']:,.0f}")

    st.markdown(f"""
| Parameter | Value |
|---|---|
| Stop Loss (1.5× ATR) | {cur_sym}{kelly['stop_1x']:,.2f} |
| Stop Loss (2.0× ATR) | {cur_sym}{kelly['stop_2x']:,.2f} |
| Target 1 (1:1 R:R)   | {cur_sym}{kelly['target_1r']:,.2f} |
| Target 2 (2:1 R:R)   | {cur_sym}{kelly['target_2r']:,.2f} |
| Approx. Shares       | {kelly['n_shares']} |
""")
    st.caption("⚠ Mathematical framework only — not financial advice.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — PEERS
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    method_icon = "🤖 LLM-discovered" if peer_method == "LLM" else "📚 Sector database"
    st.subheader(f"Sector Peer Comparison — {raw['sector']}")
    st.caption(f"Peer discovery: **{method_icon}** · {len(auto_peers)} companies")
    if not df_peers.empty:
        # Highlight the target ticker row
        st.dataframe(df_peers, use_container_width=True)

        tab_p1, tab_p2, tab_p3 = st.tabs(["P/E Comparison", "Market Cap", "ROE vs P/B"])
        with tab_p1:
            if "P/E" in df_peers.columns:
                df_pe = df_peers["P/E"].dropna().reset_index()
                fig_pe = px.bar(df_pe, x="Ticker", y="P/E",
                                 color="P/E", color_continuous_scale=["#1a5276","#2e86c1","#5dade2"],
                                 title="P/E Ratio Comparison")
                fig_pe.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d1117")
                st.plotly_chart(fig_pe, use_container_width=True)
        with tab_p2:
            if "Mkt Cap (Cr)" in df_peers.columns:
                df_mc = df_peers["Mkt Cap (Cr)"].dropna().reset_index()
                fig_mc = px.bar(df_mc, x="Ticker", y="Mkt Cap (Cr)",
                                 color="Mkt Cap (Cr)", color_continuous_scale="blues",
                                 title="Market Cap Comparison (₹ Cr)")
                fig_mc.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d1117")
                st.plotly_chart(fig_mc, use_container_width=True)
        with tab_p3:
            if "ROE (%)" in df_peers.columns and "P/B" in df_peers.columns:
                df_rb = df_peers[["ROE (%)","P/B","Mkt Cap (Cr)"]].dropna().reset_index()
                fig_rb = px.scatter(df_rb, x="P/B", y="ROE (%)", text="Ticker",
                                     size="Mkt Cap (Cr)" if "Mkt Cap (Cr)" in df_rb else None,
                                     color="ROE (%)", color_continuous_scale="RdYlGn",
                                     title="ROE (%) vs P/B — bubble = market cap")
                fig_rb.update_traces(textposition="top center")
                fig_rb.update_layout(template="plotly_dark", height=380, paper_bgcolor="#0d1117")
                st.plotly_chart(fig_rb, use_container_width=True)
    else:
        st.warning("Could not fetch peer data.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 8 — THESIS
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("🤖 AI Investment Thesis")
    st.markdown(thesis)
    st.divider()
    st.caption("⚠ This is AI-generated analysis for informational purposes only. "
               "It does not constitute financial advice. Always do your own research.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"SENTRAL Multi-Spectrum Stock Analysis · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · "
           f"For informational purposes only")
