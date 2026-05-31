# SENTRAL × SCREENER — Multi-Spectrum Stock Analysis Platform

> **An end-to-end, open-source stock analysis platform combining deep fundamental scoring, 35-indicator technical analysis, multi-source news sentiment (10 LLMs), LSTM + Transformer price forecasting, 20-strategy backtesting, and a full-featured screener inspired by screener.in × groww.in — all wrapped in two production-ready Streamlit apps.**

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Project Architecture](#project-architecture)
3. [SENTRAL — Complete Stock Analysis](#sentral--complete-stock-analysis)
   - [Architecture Overview](#architecture-overview)
   - [Notebook Guide (82 cells)](#sentral_completeipynb--notebook-guide-82-cells)
   - [Pillar 1 — Fundamentals](#pillar-1--fundamental-analysis)
   - [Pillar 2 — Technical Analysis (35 indicators)](#pillar-2--technical-analysis-35-indicators)
   - [Pillar 3 — News Aggregation (13 sources)](#pillar-3--news-aggregation-13-sources)
   - [Pillar 4 — Sentiment Analysis (10 models)](#pillar-4--sentiment-analysis-10-models)
   - [Pillar 5 — ML Price Forecasting](#pillar-5--ml-price-forecasting)
   - [Pillar 6 — Signal Engine](#pillar-6--signal-engine)
   - [Pillar 7 — 20-Strategy Backtesting](#pillar-7--20-strategy-backtesting)
   - [Pillar 8 — AI Investment Thesis](#pillar-8--ai-investment-thesis)
4. [SENTRAL App — Tab Guide](#sentral-app--tab-by-tab-guide)
5. [Module Reference (sentral_app)](#module-reference--sentral_app)
6. [SCREENER — screener.in × groww.in Replica](#screener--fundamental--technical-deep-dive)
   - [18-Section Notebook Guide](#18-section-notebook-guide)
   - [SCREENER App — 11-Tab Guide](#screener-app--11-tab-guide)
7. [Streamlit Apps — Quick Launch](#streamlit-apps)
8. [LLM-Powered Peer Discovery](#llm-powered-peer-discovery)
9. [Reddit Scraping — OAuth vs Anonymous](#reddit-scraping--oauth-vs-anonymous)
10. [API Keys Reference](#api-keys-reference)
11. [Output Files](#output-files)
12. [Quick Start](#quick-start)
13. [Technology Stack](#technology-stack)

> **Deep-dive documentation:**  
> - Full SENTRAL pipeline → [SENTRAL.md](SENTRAL.md)  
> - Full SCREENER pipeline → [SCREENER.md](SCREENER.md)  
> - Setup & troubleshooting → [SETUP.md](SETUP.md)

---

## Repository Overview

| Item | Description |
|------|-------------|
| `SENTRAL_Complete.ipynb` | 82-cell notebook: full multi-spectrum analysis pipeline |
| `screener.ipynb` | 60+ cell notebook: screener.in × groww.in replica |
| `sentral_app/` | Streamlit app for SENTRAL (8 tabs, ML forecasting) |
| `screener_app/` | Streamlit app for SCREENER (11 tabs, multi-stock screener) |
| `outputs/` | Per-run output folder (models, CSVs, JSON reports) |
| `requirements.txt` | Unified Python dependencies |
| `SENTRAL.md` | **Deep-dive docs for the SENTRAL pipeline** — notebook guide, module API, signal engine, backtest strategies, Reddit OAuth, outputs reference |
| `SCREENER.md` | **Deep-dive docs for the SCREENER** — all 18 notebook sections, 11-tab app guide, valuation model formulas, LLM peer discovery detail |
| `SETUP.md` | Step-by-step setup guide, venv, `.env`, troubleshooting |

---

## Project Architecture

```
SENTRAL-Multi-Spectrum-Stock-Analysis/
│
├── SENTRAL_Complete.ipynb          ← 82-cell analysis pipeline
├── screener.ipynb                  ← 60-cell screener.in × groww.in replica
├── BestBasic.ipynb                 ← Lightweight quick-analysis notebook
├── complete.ipynb                  ← Earlier experimental notebook
│
├── sentral_app/                    ← SENTRAL Streamlit app
│   ├── app.py                      ← Main entry point (~870 lines)
│   ├── requirements_app.txt
│   └── modules/
│       ├── data_loader.py          ← yfinance download + 35 TA indicators
│       ├── fundamental.py          ← DCF, Altman Z, Piotroski, ratios
│       ├── technical.py            ← Risk metrics, patterns, seasonality
│       ├── peers.py                ← LLM-first peer discovery
│       ├── news.py                 ← 13 news sources (APIs + RSS + Reddit)
│       ├── sentiment.py            ← 10 sentiment models
│       ├── ml_forecast.py          ← LSTM + Transformer + Monte Carlo
│       ├── signals.py              ← Composite BUY/HOLD/SELL signal
│       ├── backtest.py             ← 20-strategy backtesting engine
│       └── report.py               ← HTML + PDF report generation
│
├── screener_app/                   ← SCREENER Streamlit app
│   ├── app.py                      ← Main entry point (~520 lines, 11 tabs)
│   ├── requirements_app.txt
│   └── modules/
│       ├── data_loader.py          ← Full yfinance data download + TA
│       ├── fundamentals.py         ← P&L, BS, CF, ratios, DCF, Graham
│       ├── peers.py                ← LLM-first peer discovery
│       └── screener.py             ← Multi-stock screener engine
│
├── outputs/
│   └── TICKER_YYYYMMDD_HHMM/
│       ├── backtest.csv
│       ├── forecast_30d.csv
│       ├── lstm_model.pt
│       ├── news_corpus.csv
│       ├── price_indicators.csv
│       ├── sentiment_summary.csv
│       ├── signal_report.json
│       └── transformer_model.pt
│
├── requirements.txt                ← Unified dependencies
├── SETUP.md                        ← Step-by-step setup guide
├── SENTRAL.md                      ← Deep-dive docs: SENTRAL pipeline
└── SCREENER.md                     ← Deep-dive docs: SCREENER pipeline
```

---

## SENTRAL — Complete Stock Analysis

> **Full deep-dive documentation:** [SENTRAL.md](SENTRAL.md) — covers every notebook cell, all module APIs, signal math, backtest strategy conditions, Reddit OAuth upgrade guide, and the full outputs reference.

SENTRAL (**Sen**timent + **T**echnical + fund**a**mental Ana**l**ysis) is an 8-pillar end-to-end stock analysis system — available as an **82-cell Jupyter notebook** (`SENTRAL_Complete.ipynb`) and an **8-tab Streamlit app** (`sentral_app/`).

---

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        SENTRAL Pipeline                         │
│  Ticker Input                                                    │
│       │                                                          │
│       ├──→ [1] Data Loader    (yfinance OHLCV + 35 TA cols)     │
│       ├──→ [2] Fundamental    (38 metrics, Altman Z, Piotroski)  │
│       ├──→ [3] Technical      (35 indicators + risk metrics)     │
│       ├──→ [4] News           (13 sources — 7 APIs + 6 RSS)      │
│       ├──→ [5] Sentiment      (10 models, weighted ensemble)     │
│       ├──→ [6] ML Forecast    (LSTM + Transformer + Monte Carlo) │
│       │                                                          │
│       ├──→ [7] Signal Engine                                     │
│       │         └── 0.30×F + 0.30×T + 0.20×S + 0.20×M          │
│       │             → BUY / HOLD / SELL + Confidence %          │
│       │                                                          │
│       ├──→ [8] Backtesting    (20 strategies, full metrics)      │
│       └──→ [9] Report         (HTML + PDF + JSON)                │
└────────────────────────────────────────────────────────────────┘
```

| Pillar | Weight | Rationale |
|--------|--------|-----------|
| Fundamental | 0.30 | Long-term business quality is the foundation of intrinsic value |
| Technical | 0.30 | Price action reflects market consensus and momentum |
| Sentiment | 0.20 | News/social tone drives near-term volatility |
| Forecast (ML) | 0.20 | Forward-looking context with measured uncertainty |

---

### `SENTRAL_Complete.ipynb` — Notebook Guide (82 Cells)

Run with kernel: *Python (SENTRAL venv)*. Configuration in Cell 2:

```python
TICKER         = "HAL.NS"    # Any Yahoo Finance symbol (NSE .NS / BSE .BO / US)
PERIOD         = "5y"        # Historical data: 1y / 2y / 3y / 5y / 10y / max
BENCH_TICKER   = "^NSEI"     # Benchmark: ^NSEI (Nifty 50), ^GSPC (S&P 500)
FORECAST_DAYS  = 30          # Days to forecast forward
LOOKBACK       = 60          # LSTM sequence window length (days)
N_SIMULATIONS  = 1000        # Monte Carlo simulation paths
USE_ML         = True        # Toggle ML forecasting (disable for faster runs)
```

| Cell Range | Pillar | Content |
|------------|--------|---------|
| 1–5 | Setup | Imports, config, API keys, output directory |
| 6–18 | Fundamentals | 38 metrics, Altman Z, Piotroski, DCF, Graham, peers |
| 19–26 | Technical | 35 indicators, 6-chart dashboard, risk metrics |
| 27–32 | News | 13-source fetch, relevance scoring, timeline |
| 33–42 | Sentiment | 10-model inference, ensemble, distribution charts |
| 43–54 | ML Forecast | LSTM + Transformer training, Monte Carlo, forecast chart |
| 55 | Signal | Composite score computation + display |
| 56–68 | Backtest | 20 strategies, equity curves, metrics table |
| 69–82 | Report | HTML/PDF report, JSON dump, AI thesis |

---

### Pillar 1 — Fundamental Analysis

**38 metrics across 5 categories:**

| Category | Metrics |
|----------|---------|
| Valuation | P/E (trailing + forward), P/B, P/S, EV/EBITDA, EV/Revenue, PEG Ratio |
| Profitability | Gross Margin, Operating Margin, EBITDA Margin, Net Margin, ROE, ROA |
| Growth | Revenue Growth YoY, Earnings Growth YoY, FCF Growth |
| Solvency | D/E Ratio, Current Ratio, Quick Ratio, Interest Coverage, Total Debt (Cr) |
| Per-Share | EPS (TTM), Book Value/Share, Revenue/Share, Dividend Yield, Dividend/Share |

**Altman Z-Score:**
$$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5$$

| Variable | Formula | Measures |
|----------|---------|---------|
| X₁ | Working Capital / Total Assets | Short-term liquidity |
| X₂ | Retained Earnings / Total Assets | Accumulated profitability |
| X₃ | EBIT / Total Assets | Operating efficiency |
| X₄ | Market Cap / Total Liabilities | Leverage buffer |
| X₅ | Revenue / Total Assets | Asset utilisation |

| Z-Score | Zone |
|---------|------|
| > 2.99 | Safe ✅ |
| 1.81 – 2.99 | Grey ⚠️ |
| < 1.81 | Distress 🔴 |

> **Indian stock note:** `yfinance` returns `None` for `info.get("totalAssets")` on NSE/BSE stocks. SENTRAL reads `Total Assets` directly from the balance sheet DataFrame.

**Piotroski F-Score (9 criteria):**

| Group | Criterion | +1 if… |
|-------|-----------|--------|
| Profitability | ROA > 0 | Positive return on assets |
| | OCF > 0 | Positive operating cash flow |
| | ROA improving | Higher ROA vs prior year |
| | Accruals (OCF/TA > ROA) | Cash earnings exceed reported earnings |
| Leverage | LT Debt decreasing | Less long-term debt than prior year |
| | Current Ratio improving | Better liquidity than prior year |
| | No dilution | Share count not increased |
| Efficiency | Gross Margin improving | Higher gross margin than prior year |
| | Asset Turnover improving | Better asset utilisation |

**Score ≥ 7 = Strong 🟢 · 4–6 = Moderate 🟡 · < 4 = Weak 🔴**

---

### Pillar 2 — Technical Analysis (35 Indicators)

| Category | Indicators |
|----------|------------|
| Moving Averages | SMA 10/20/50/100/200, EMA 10/20/50/200 |
| Momentum | RSI-14, Stochastic %K/%D, Williams %R, ROC-10 |
| Trend | MACD, MACD Signal, MACD Hist, ADX, +DI/−DI, CCI, DPO, Ichimoku A/B |
| Volatility | Bollinger Bands (±2σ), BB%, ATR, Keltner Upper/Lower, Ulcer Index |
| Volume | OBV, CMF (Chaikin Money Flow), MFI, VWAP |
| Returns | 1-day, 5-day, 20-day simple returns |

**Risk metrics:**

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | `(CAGR − 6%) / annualised_std` |
| Sortino Ratio | `(CAGR − 6%) / downside_deviation` |
| VaR (95%) | 5th percentile of daily log returns |
| CVaR (ES) | Mean of returns below VaR |
| Max Drawdown | Peak-to-trough decline over full period |
| Calmar Ratio | `CAGR / |Max Drawdown|` |

---

### Pillar 3 — News Aggregation (13 Sources)

| Source | Type | Coverage |
|--------|------|---------|
| Alpha Vantage | REST API | Global + pre-labeled sentiment |
| Finnhub | REST API | Company-specific news |
| Tavily | AI Search API | Deep web + LLM-summarised |
| NewsAPI | REST API | 70,000+ global sources |
| EODHD | REST API | Financial + filing news |
| Marketaux | REST API | Market-sentiment headlines |
| APILayer | REST API | Financial headline feed |
| Yahoo Finance | RSS | Real-time Yahoo Finance articles |
| Google News | RSS | Broad news via Google RSS |
| ET Markets | RSS | India (Economic Times) |
| Livemint | RSS | India business news |
| Reddit | RSS → OAuth upgrade | Retail investor sentiment |
| StockTwits | RSS | Stock-specific social sentiment |

Articles are **relevance-scored** by ticker/company name mention frequency; only `relevance_score ≥ 1` articles pass to sentiment analysis.

---

### Pillar 4 — Sentiment Analysis (10 Models)

| Model | Architecture | Specialisation | Weight |
|-------|-------------|----------------|--------|
| NLTK VADER | Rule-based lexicon | General text | 0.8× |
| FinBERT (ProsusAI) | BERT-base | Financial news | 1.5× |
| FinBERT-Tone (yiyanghkust) | BERT-base | Tone analysis | 1.5× |
| DistilRoBERTa | DistilRoBERTa | Fast general sentiment | 1.0× |
| RoBERTa-Large | RoBERTa-large | High-accuracy general | 1.2× |
| twitter-roberta (cardiffnlp) | RoBERTa | Short-form social text | 1.0× |
| Groq llama-3.3-70b-versatile | LLaMA 3.3 70B | Reasoning + context | 2.0× |
| Groq llama-4-scout | LLaMA 4 | Multi-modal reasoning | 2.0× |
| Groq qwen-32b | Qwen 32B | Multi-lingual | 2.0× |
| Gemini 2.5 Flash | Gemini | Long-context analysis | 2.0× |

**Groq cascade:** `llama-3.3-70b-versatile` → `llama-4-scout` → `llama-3.1-8b-instant` → `qwen-32b-preview` → `deepseek-r1-distill-llama-70b` → `mistral-saba-24b`

**Gemini cascade:** `gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash`

**Ensemble formula:**
$$S_{ensemble} = \frac{\sum_i w_i \cdot s_i}{\sum_i w_i}, \quad s_i \in \{-1,\, 0,\, +1\}$$

Label: **Bullish** > +0.15 · **Neutral** −0.15 to +0.15 · **Bearish** < −0.15

---

### Pillar 5 — ML Price Forecasting

#### LSTM Architecture
```
Input  (batch, seq_len=60, features=1)
  → LSTM Layer 1  (hidden=128, dropout=0.2)
  → LSTM Layer 2  (hidden=128, dropout=0.2)
  → Linear(128→64) → ReLU → Linear(64→1)
Output: predicted next-day close price
```
80/20 split · MinMaxScaler [0,1] · Adam lr=0.001 · MSE loss · 50 epochs · saved as `lstm_model.pt`

#### Temporal Transformer Architecture
```
Input → Linear(1, d_model=64) → Positional Encoding (sinusoidal)
  → TransformerEncoder (nhead=8, layers=2, dim_feedforward=256, dropout=0.1)
  → Global Average Pooling → Linear(64→1)
Output: predicted next-day close price
```
Saved as `transformer_model.pt`

#### Monte Carlo GBM
$$S_{t+1} = S_t \cdot \exp\!\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon\right), \quad \varepsilon \sim \mathcal{N}(0,1)$$

- **μ** = annualised mean log return · **σ** = annualised std of log returns
- Output: P10/P50/P90 percentile paths · P(profit > 0) · Forecast VaR

#### Ensemble
```
Ensemble[t] = 0.5 × LSTM[t] + 0.5 × Transformer[t]
Score       = clip((Ensemble_30d_return − μ) / σ, −1, +1)
```

---

### Pillar 6 — Signal Engine

$$\text{Score} = 0.30 \cdot F + 0.30 \cdot T + 0.20 \cdot S + 0.20 \cdot M \quad \in [-1, +1]$$

| Variable | Source | Range |
|----------|--------|-------|
| F | Fundamental score (38 metrics vs sector medians) | [−1, +1] |
| T | Technical score (35-indicator directional vote) | [−1, +1] |
| S | Sentiment ensemble (10-model weighted average) | [−1, +1] |
| M | ML forecast score (30-day return, normalised) | [−1, +1] |

| Score | Signal |
|-------|--------|
| > +0.15 | BUY 🟢 |
| −0.15 to +0.15 | HOLD 🟡 |
| < −0.15 | SELL 🔴 |

**Risk flags** (up to 5 warnings displayed):
- Altman Z < 1.81 · D/E > 3.0 · RSI > 80 or < 20 · Max Drawdown > 40% · Daily VaR > 5%

---

### Pillar 7 — 20-Strategy Backtesting

**Assumptions:** 0.1% commission round-trip · 100% equity · close-price execution

| # | Strategy | Entry Condition |
|---|----------|----------------|
| 1 | EMA + MACD + RSI | EMA cross AND MACD bullish AND RSI 30–70 |
| 2 | Golden / Death Cross | SMA50 crosses SMA200 |
| 3 | MACD Signal Cross | MACD line crosses Signal |
| 4 | RSI Oversold Bounce | RSI crosses up through 30 |
| 5 | RSI Overbought Exit | RSI crosses down through 70 |
| 6 | Bollinger Mean Reversion | Price touches lower/upper band |
| 7 | ADX Trend Following | ADX > 25 AND +DI > −DI |
| 8 | Stochastic Crossover | %K × %D in oversold/overbought zone |
| 9 | Williams %R | %R crosses −80 (buy) / −20 (sell) |
| 10 | CMF Positive Flow | CMF crosses 0 from below |
| 11 | OBV Momentum | OBV > 20-day OBV SMA |
| 12 | VWAP Cross | Price crosses VWAP |
| 13 | MFI Signal | MFI crosses 20 (buy) / 80 (sell) |
| 14 | ROC Signal | Rate-of-Change crosses 0 |
| 15 | Triple EMA | EMA9 > EMA21 > EMA55 |
| 16 | Donchian Channel | Price breaks 20-day high/low |
| 17 | Keltner Breakout | Price breaks Keltner channel |
| 18 | Multi-Confluence | ≥ 5 indicators agree on direction |
| 19 | Ichimoku Cloud | Price above/below cloud + Tenkan/Kijun cross |
| 20 | Buy & Hold | Always long (baseline) |

**Output per strategy** (→ `backtest.csv`): Total Return · CAGR · Sharpe · Sortino · Max Drawdown · Win Rate · Profit Factor · Trades · Kelly Criterion · Stop-Loss suggestion

---

### Pillar 8 — AI Investment Thesis

Uses Gemini cascade (`gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash`) with a fully structured prompt containing all pillar outputs:

**Thesis output sections:**
1. **Company Summary** — business description and competitive moat
2. **Bull Case** — 3 data-driven reasons to invest
3. **Bear Case** — 3 key risks and concerns
4. **Valuation Assessment** — DCF vs Graham vs current market price
5. **Technical Outlook** — trend, momentum, key support/resistance
6. **Sentiment Context** — news tone, social signal strength
7. **Recommendation** — BUY/HOLD/SELL with confidence and suggested time horizon

The thesis is rendered as formatted HTML in the app and embedded in the PDF report.

---

## SENTRAL App — Tab-by-Tab Guide

```bash
cd sentral_app && streamlit run app.py
```

**Sidebar controls:** Ticker · Period · Forecast Days (7–120) · LSTM Lookback (20–120) · Monte Carlo N (100–5000) · ML toggle · 10 API key fields (pre-loaded from `.env`) · **Upload Bundle ZIP** · **⚡ Load from Bundle**

**Download buttons (above tabs after any analysis):**

| Button | Output |
|--------|--------|
| ⬇ HTML Report | Full interactive report |
| ⬇ PDF Report | Static PDF (`reportlab` required) |
| ⬇ Price + Indicators CSV | OHLCV + 35 TA columns |
| 📦 Download Full Bundle ZIP | Complete ZIP (news, sentiment, ML predictions, backtest curves, scalars) — upload back to skip 5–10 min pipeline |

| Tab | Contents |
|-----|----------|
| **📊 Signal Engine** | Composite score gauge, BUY/HOLD/SELL banner, 4 pillar score bars, confidence %, risk flags, signal download |
| **🔢 Fundamentals** | 38-metric display (5 tables), DCF card, Graham Number card, Altman Z gauge, Piotroski F scorecard, P&L + BS + CF charts |
| **📈 Technical** | 6-row chart (Price+MA, RSI, MACD, Stochastic, BB, Volume), Ichimoku cloud, risk metrics table |
| **📰 News & Sentiment** | Relevance-sorted article feed, per-model sentiment scores, ensemble gauge, model agreement chart |
| **🤖 ML Forecast** | 30-day LSTM+Transformer ensemble chart, Monte Carlo fan (P10/P50/P90), P(profit) metric, training curves |
| **⚙️ Backtest** | 20-strategy results table (colour-coded), top-5 equity curves, best strategy highlight |
| **👥 Peers** | LLM discovery badge, 12-column peer table, P/E bar, Market Cap bar, ROE vs P/B scatter |
| **🧠 AI Thesis** | Full Gemini thesis (8 sections), PDF export button |

---

## Module Reference — `sentral_app`

| Module | Key Functions |
|--------|--------------|
| `data_loader.py` | `download_stock_data(ticker, period) → dict` · `compute_features(df) → df` |
| `fundamental.py` | `compute_fundamental_score(info, bal, fin, cf) → float` · `compute_altman_z(bal, fin) → (float, str)` · `compute_piotroski(bal, fin, cf) → (int, dict)` · `compute_dcf(cf, info, wacc, g) → dict` |
| `technical.py` | `compute_technical_score(data) → float` · `compute_risk_metrics(data) → dict` · `detect_patterns(data) → list` |
| `news.py` | `fetch_all_news(ticker, company, keys) → list[dict]` |
| `sentiment.py` | `SentimentAnalyzer.analyze_batch(articles) → dict` · `ensemble_score(scores) → (float, str)` |
| `ml_forecast.py` | `train_lstm(data, lookback, ...) → (model, scaler)` · `train_transformer(...)` · `ensemble_forecast(...)` · `monte_carlo_simulation(...)` |
| `signals.py` | `compute_composite_signal(F, T, S, M, weights) → {signal, score, confidence, flags}` |
| `backtest.py` | `run_all_strategies(data) → pd.DataFrame` · `get_optimal_strategy(results) → str` |
| `peers.py` | `get_peer_data(ticker, sector, company, exchange, groq_k, gemini_k) → (list, df, method)` |
| `report.py` | `generate_html_report(ticker, data, dir) → path` · `generate_pdf_report(ticker, data, dir) → path` |

> See [SENTRAL.md → Module Deep-Dives](SENTRAL.md#module-deep-dives) for full function signatures and return types.

---

## Sample `signal_report.json`

```json
{
  "ticker": "HAL.NS",
  "company": "Hindustan Aeronautics Limited",
  "timestamp": "2024-01-15T10:30:00",
  "price": 4303.80,
  "signal": "HOLD",
  "confidence": 0.42,
  "composite_score": 0.08,
  "pillar_scores": {
    "fundamental": 0.35,
    "technical": -0.12,
    "sentiment": 0.15,
    "forecast": -0.10
  },
  "fundamentals": {
    "pe": 31.55, "pb": 7.01, "roe": 23.98, "debt_equity": 0.03,
    "altman_z": 2.99, "altman_zone": "Safe",
    "piotroski": 8, "piotroski_signal": "Strong",
    "dcf_intrinsic": 2083.01, "graham_number": 1372.48
  },
  "technical": {
    "rsi": 43.7, "macd_signal": "Bearish",
    "bb_pct": 0.19, "trend": "Sideways"
  },
  "sentiment": {
    "ensemble_score": 0.15, "label": "Slightly Bullish",
    "article_count": 42, "model_count": 7
  },
  "forecast": {
    "lstm_30d_return": -2.1,
    "transformer_30d_return": -3.4,
    "ensemble_30d_return": -2.75,
    "monte_carlo_p_profit": 0.44
  },
  "best_backtest_strategy": "EMA_MACD_RSI",
  "best_backtest_cagr": 38.2
}
```

---

## SCREENER — Fundamental + Technical Deep Dive

> **Full documentation:** [SCREENER.md](SCREENER.md). This section covers all 18 notebook sections, the 11-tab Streamlit app, valuation formulas, and the screener engine inline.

SCREENER is a **Python implementation of screener.in × groww.in** — covering company overview, quarterly/annual financials, 25+ ratios, shareholding, LLM peer discovery, valuation models, technical indicators, a multi-stock screener, and an investment calculator.

### Feature Coverage

| Feature | screener.in ✓ | groww.in ✓ |
|---------|:---:|:---:|
| Quarterly P&L table | ✅ | ✅ |
| Annual P&L + balance sheet | ✅ | ✅ |
| Cash flow statement | ✅ | — |
| Financial ratio analysis | ✅ | ✅ |
| Shareholding pattern | ✅ | ✅ |
| Peer comparison table | ✅ | ✅ |
| Custom stock screener | ✅ | — |
| Investment calculator (SIP) | — | ✅ |
| Technical chart | — | ✅ |
| Valuation models (DCF, Graham) | ✅ | — |
| Piotroski F-Score | ✅ | — |
| Altman Z-Score | ✅ | — |
| Dividend history | ✅ | ✅ |
| Corporate events calendar | ✅ | ✅ |
| OBV + Volume indicators | — | ✅ |
| Pros & Cons analysis | ✅ | ✅ |

---

### `screener.ipynb` — 18 Sections

Configure in **Cell 2**:
```python
TICKER = "HAL.NS"   # Any Yahoo Finance symbol
PERIOD = "5y"       # 1y / 2y / 3y / 5y / 10y / max
BENCH  = "^NSEI"    # Benchmark: ^NSEI, ^GSPC, ^BSE200
```
Loads `GROQ_API_KEY` + `GEMINI_API_KEY` from `.env` for peer discovery.

**Cell 3** downloads: OHLCV history · Annual + quarterly financials (P&L, balance sheet, cash flow) · Dividends + splits · Institutional holders · Earnings calendar · Benchmark history

| Section | Cells | Content |
|---------|-------|---------|
| **1. Company Overview** | 4–6 | Header card (price, 52W range, market cap, 18 key stats), 12+ auto-generated Pros 🟢 / Cons 🔴 |
| **2. Price Chart** | 7–9 | Plotly candlestick + SMA20/50/200 + EMA20 overlays; period returns vs benchmark (1M/3M/6M/1Y/3Y) |
| **3. Quarterly Results** | 10–11 | 12-quarter table (Revenue, OP Inc, Net Profit, OPM%, NPM%, EPS) with margin gradients; 4-subplot chart |
| **4. Annual P&L** | 12–13 | Annual table + grouped bars (Revenue/Profit) + margin trend lines (Gross/Operating/Net) |
| **5. Balance Sheet** | 14–15 | Annual table + Assets vs Liabilities vs Equity grouped bar; D/E + Quick Ratio trends |
| **6. Cash Flow** | 16–17 | OCF / Investing CF / Financing CF / FCF (OCF − CapEx) grouped bar chart |
| **7. Financial Ratios** | 18–20 | 4 categories (Valuation/Profitability/Solvency/Per-Share), 25+ ratios; radar scorecard |
| **8. Shareholding Pattern** | 21–22 | Donut pie (Promoter / Institutional / Public); top 15 institutional holders table |
| **9. Peer Comparison** | 23–26 | LLM-discovered peers; 12-column styled table; 4-chart grid + ROE vs P/E bubble scatter |
| **10. Technical Analysis** | 27–29 | Signals summary table (RSI/MACD/BB/ADX/Stochastic/OBV/Ichimoku); 4-row Plotly chart |
| **11. Dividend History** | 30–31 | Per-event + annual totals bar charts; total paid, average annual yield |
| **12. Valuation Models** | 32–33 | DCF, Graham Number, P/E Mean Reversion, Lynch PEG — side-by-side comparison |
| **13. Multi-Stock Screener** | 34–35 | 60 NSE stocks, 6 configurable filters, colour-graded results table |
| **14. Corporate Events** | 36–37 | Next earnings date + estimates; ex-dividend date; splits history |
| **15. OBV + Volume** | 38–39 | 3-row chart: OBV / CMF (Chaikin Money Flow) / MFI (Money Flow Index) |
| **16. Investment Calculator** | 40–41 | SIP + lump-sum projector; historical CAGR pre-filled; 10-year growth chart |
| **17. Financial Health** | 42–43 | Piotroski F-Score (9 criteria table) + Altman Z-Score; zone interpretation |
| **18. Company Profile** | 44–45 | Full description, HQ, employees, website, CEO and officers list |

---

### Valuation Models (Section 12)

**1. DCF — Discounted Cash Flow**
$$\text{Intrinsic} = \sum_{n=1}^{10} \frac{FCF_n}{(1+\text{WACC})^n} + \frac{TV}{(1+\text{WACC})^{10}}$$
- Base FCF = average of last 4 years' Free Cash Flow
- Growth projected at revenue growth rate, capped at 20%
- Terminal Value: `TV = FCF₁₀ × (1+g) / (WACC − g)`

**2. Graham Number**
$$\text{Graham} = \sqrt{22.5 \times \text{EPS} \times \text{BookValuePerShare}}$$
Benjamin Graham's conservative intrinsic value — assumes fair stock satisfies P/E × P/B ≤ 22.5.

**3. P/E Mean Reversion**
$$\text{FairValue} = \text{EPS} \times \text{Sector Median P/E}$$
Sector PE lookup: Technology=35 · Financials=18 · Healthcare=28 · Industrials=24 · Energy=12 · default=22

**4. Lynch PEG Ratio**
$$\text{PEG} = \frac{P/E}{\text{EPS Growth Rate}}$$
PEG < 1 → potentially undervalued · 1–2 → fair · > 2 → potentially overvalued

---

### SCREENER App (`screener_app/`) — 11 Tabs

```bash
cd screener_app && streamlit run app.py
```
**Sidebar:** Ticker · Period · Benchmark · Groq API Key · Gemini API Key

| Tab | Contents |
|-----|----------|
| **Overview** | 12-metric grid card, 52W range progress bar, Pros 🟢 / Cons 🔴 auto-analysis, company description |
| **Price Chart** | Candlestick/Line/OHLC toggle, MA selector (SMA20/50/200, EMA20/50), period returns vs benchmark |
| **Financials** | 3 sub-tabs: P&L (margins + bars) · Balance Sheet (assets/liabilities/equity) · Cash Flow (OCF/ICF/FCF) |
| **Ratios** | 4-category tables in 2×2 grid, radar scorecard, Altman Z, Piotroski F (all 9 criteria in expander) |
| **Quarterly** | 12-quarter gradient table, 4-subplot chart (Revenue / Net Profit / OPM% / EPS) |
| **Shareholding** | Promoter/Institutional/Public donut pie, top 15 institutional holders |
| **Peers** | Discovery badge (🤖 LLM or 📚 DB), 12-column styled table, P/E bars, Mkt Cap bars, ROE vs P/B scatter |
| **Valuation** | DCF (WACC 8–20%, growth 2–8% sliders), Graham Number, P/E Reversion — all side-by-side |
| **Technical** | RSI/MACD/ADX/ATR metric cards; 4-row chart (Price+BB / RSI / MACD / Stochastic); OBV+CMF |
| **Screener** | 8 filter sliders, 60-stock NSE universe, colour-graded results table, CSV download |
| **Calculator** | SIP + lump-sum comparison; historical CAGR pre-filled; 10-year growth chart |

---

### Multi-Stock Screener Engine

**Default universe:** 60 NSE stocks across 10 sectors (IT · Banking · Auto · Pharma · Energy · FMCG · Metals · Infra · Defence · Consumer)

**Available filters:**

| Filter | Operator | Default |
|--------|----------|---------|
| P/E Ratio | ≤ | 30 |
| Price-to-Book | ≤ | 5 |
| ROE % | ≥ | 15 |
| Net Margin % | ≥ | 10 |
| Debt/Equity | ≤ | 1.0 |
| Market Cap (₹ Cr) | ≥ | 50,000 |
| Revenue Growth % | ≥ | 10 |
| Dividend Yield % | ≥ | 0 |

Fetched metrics per stock: Name · Sector · Price · Mkt Cap · P/E · P/B · ROE% · Net Margin% · D/E · Rev Growth% · Div Yield% · Current Ratio · Beta · EPS

---

### SCREENER Module Reference

| Module | Key Functions |
|--------|--------------|
| `data_loader.py` | `download_full_data(ticker, period) → dict` (22 keys, all pickle-safe) · `compute_technical_indicators(hist) → df` (25+ TA columns) |
| `fundamentals.py` | `build_pl_table(fin, info, cur)` · `build_balance_sheet(bal, cur)` · `build_cashflow(cf, cur)` · `compute_all_ratios(info) → dict` · `compute_piotroski_fscore(bal, fin, cf, info) → dict` · `compute_altman_z(info) → dict` · `compute_dcf(cf, info, wacc, tg) → dict` · `compute_graham_number(info) → float` |
| `peers.py` | `get_peer_data(ticker, sector, company, exchange, groq_k, gemini_k) → (list, df, method)` — 3-tier: Groq → Gemini → sector DB |
| `screener.py` | `DEFAULT_UNIVERSE: list[str]` (60 tickers) · `fetch_screener_data(tickers: tuple) → df` (`@st.cache_data(ttl=1800)`) · `apply_filters(df, filters) → df` |

---

## Streamlit Apps

### SENTRAL App (`sentral_app/`)
```bash
cd sentral_app && streamlit run app.py
```

See [SENTRAL App — Tab-by-Tab Guide](#sentral-app--tab-by-tab-guide) above for full details.

**8 tabs:** Signal Engine | Fundamentals | Technical | News & Sentiment | ML Forecast | Backtest | Peers | AI Thesis

### SCREENER App (`screener_app/`)
```bash
cd screener_app && streamlit run app.py
```

| Tab | Contents |
|-----|----------|
| **Overview** | Header card (price, change, market cap, 52W), key stats table (18 metrics), Strengths & Weaknesses (pros/cons/neutral) |
| **Price Chart** | Interactive candlestick, EMA 20/50/200 overlays, volume bars, period returns vs Nifty 50 benchmark, drawdown chart |
| **Financials** | Annual P&L trends (Revenue, EBITDA, PAT, margins), Annual Cash Flow (OCF/ICF/FCF), Annual Balance Sheet (assets vs liabilities) |
| **Ratios** | 4-category ratio tables (Profitability/Valuation/Solvency/Efficiency), radar scorecard chart |
| **Quarterly** | Quarterly results table (Revenue, Gross Profit, Operating Inc, PAT, OPM%, NPM%, EPS), quarterly bar charts |
| **Shareholding** | Promoter/Institutional/Public pie chart, top 15 institutional holders table |
| **Peers** | LLM-discovered or sector-DB peer list, 14-column comparison table, bar charts for P/E/ROE/Net Margin/Mkt Cap, ROE vs P/E bubble chart |
| **Valuation** | DCF model (WACC + terminal growth sliders), Graham Number, Lynch PEG, P/E Mean Reversion, valuation summary card |
| **Technical** | Bollinger Bands chart, RSI + MACD + Stochastic in subplots, OBV + CMF + MFI volume indicators |
| **Screener** | Multi-stock screener (60 NSE stocks, 7 configurable filters), sortable results with colour gradients |
| **Calculator** | SIP + Lump-sum comparison, 10-year projection, historical CAGR estimation |

**Sidebar controls:** Ticker · Period · Benchmark · Groq/Gemini API keys · **Upload Bundle ZIP** · **⚡ Load from Bundle**

**Download buttons (above tabs after any analysis):**

| Button | Output |
|--------|--------|
| 📦 Download Analysis Bundle ZIP | Complete ZIP (financials, indicators, peers, holders, info scalars) — upload back to skip all re-downloading |
| ⬇ Price + Indicators CSV | OHLCV + all technical indicator columns |

---

## LLM-Powered Peer Discovery

Both apps and the screener notebook use a **3-tier peer discovery system**:

```
1. Groq LLM (llama-3.3-70b-versatile → llama-3.1-8b-instant)
        ↓ if no API key or fails
2. Gemini (gemini-2.5-flash → gemini-2.5-flash-lite → gemini-2.0-flash)
        ↓ if no API key or fails
3. Hard-coded sector database (12 sectors, ~8 peers each)
```

**Prompt template:**
> "List exactly 8 publicly traded peer companies for {company} ({ticker}), a {sector} company on {exchange}. Return ONLY comma-separated Yahoo Finance ticker symbols. Include {ticker} first."

Advantages of LLM discovery:
- Works for **any stock globally** — not limited to Indian/US hard-coded lists
- Captures **cross-sector conglomerates** correctly (e.g. Reliance in both Energy and Retail)
- Adapts to **company-specific competitive landscape** rather than just SIC codes
- The Streamlit apps show a **discovery method badge** (`🤖 LLM-discovered` vs `📚 Sector database`) so you know which source was used

---

## Reddit Scraping — OAuth vs Anonymous

The current implementation uses **Reddit's public RSS feed** (no authentication). This works for basic discovery but has important limitations.

### Current: Public RSS (no auth)
```python
# In sentral_app/modules/news.py
feed = feedparser.parse(f"https://www.reddit.com/search.rss?q={symbol}&limit=25")
```
- ✅ No API key required
- ✅ Works immediately out of the box
- ❌ Rate-limited to ~60 requests/hour
- ❌ Returns only top-level post titles (no comments)
- ❌ Can be blocked intermittently

### Upgrade: Reddit OAuth (PRAW)
With a Reddit developer account (free), you get:
- ✅ **60 requests/minute** per OAuth credential (60× higher rate limit)
- ✅ Full access to **post body text, comments, and upvotes**
- ✅ Access to **subreddit search** (not just global search)
- ✅ Upvote-weighted sentiment (high-karma posts carry more signal)
- ✅ Can search multiple subreddits: `r/IndianStockMarket`, `r/stocks`, `r/investing`, `r/SecurityAnalysis`
- ✅ Time-filtered queries (`after=1month`, `sort=relevance`)

**To upgrade to OAuth, install PRAW and add to `news.py`:**
```python
# pip install praw
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="SENTRAL/1.0 by u/your_username"
)

def fetch_reddit_oauth(symbol, company_name, subreddits=None, limit=25):
    if subreddits is None:
        subreddits = ["IndianStockMarket", "stocks", "investing"]
    articles = []
    for sub in subreddits:
        sr = reddit.subreddit(sub)
        for post in sr.search(f"{symbol} OR {company_name}", sort="relevance",
                               time_filter="month", limit=limit):
            articles.append({
                "title":     post.title,
                "body":      post.selftext[:500],
                "score":     post.score,
                "source":    f"Reddit r/{sub}",
                "published": datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                "url":       f"https://reddit.com{post.permalink}",
            })
    return articles
```

Register at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → create "script" type app. Add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` to your `.env` file.

---

## API Keys Reference

| Key | Service | Free Tier |
|-----|---------|-----------|
| `GROQ_API_KEY` | Groq (LLM inference) | Generous free tier |
| `GEMINI_API_KEY` | Google Gemini | Free tier available |
| `HF_TOKEN` | HuggingFace (FinBERT, etc.) | Free |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage news | 25 req/day free |
| `FINNHUB_API_KEY` | Finnhub news | 60 req/min free |
| `TAVILY_API_KEY` | Tavily AI search | 1000 req/month free |
| `NEWSAPI_KEY` | NewsAPI | 100 req/day free |
| `EODHD_API_KEY` | EODHD news | Limited free tier |
| `MARKETAUX_API_KEY` | Marketaux | 100 req/day free |
| `APILAYER_API_KEY` | APILayer news | Limited free tier |
| `REDDIT_CLIENT_ID` | Reddit OAuth (optional) | Free |
| `REDDIT_CLIENT_SECRET` | Reddit OAuth (optional) | Free |

Store all keys in a `.env` file at the repository root:
```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
HF_TOKEN=hf_...
ALPHA_VANTAGE_API_KEY=...
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SENTRAL-Multi-Spectrum-Stock-Analysis
cd SENTRAL-Multi-Spectrum-Stock-Analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env            # edit with your keys

# 5a. Run SENTRAL app
cd sentral_app
streamlit run app.py

# 5b. Run SCREENER app (new terminal)
cd screener_app
streamlit run app.py

# 5c. Or run notebooks in VS Code / JupyterLab
# Select kernel: "Python (SENTRAL venv)"
```

See [SETUP.md](SETUP.md) for detailed setup instructions including troubleshooting.

---

## Output Files

Each SENTRAL run saves to `outputs/{TICKER}_{YYYYMMDD}_{HHMM}/`:

| File | Contents |
|------|----------|
| `signal_report.json` | Composite signal, pillar scores, confidence %, all metrics |
| `price_indicators.csv` | OHLCV + all 35 technical indicators |
| `forecast_30d.csv` | LSTM + Transformer + ensemble daily forecasts |
| `backtest.csv` | All 20 strategy results (return, Sharpe, drawdown) |
| `news_corpus.csv` | All scraped news articles with relevance scores |
| `sentiment_summary.csv` | Per-model sentiment scores + ensemble |
| `lstm_model.pt` | Trained LSTM weights (PyTorch) |
| `transformer_model.pt` | Trained Transformer weights (PyTorch) |

---

## Technology Stack

| Layer | Libraries |
|-------|-----------|
| Data | `yfinance`, `pandas`, `numpy`, `scipy` |
| Technical Analysis | `ta`, `pandas-ta`, `mplfinance` |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |
| NLP/Sentiment | `nltk`, `transformers`, `torch`, `groq`, `google-generativeai` |
| ML/Forecasting | `scikit-learn`, `torch` (LSTM + Transformer) |
| News | `feedparser`, `requests`, `beautifulsoup4`, `newsapi-python` |
| Web App | `streamlit` |
| Reports | `reportlab` (PDF) |
| Environment | `python-dotenv` |

---

*SENTRAL × SCREENER — Built for research and education. Not financial advice.*
