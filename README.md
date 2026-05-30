# SENTRAL × SCREENER — Multi-Spectrum Stock Analysis Platform

> **An end-to-end, open-source stock analysis platform combining deep fundamental scoring, 35-indicator technical analysis, multi-source news sentiment (10 LLMs), LSTM + Transformer price forecasting, 20-strategy backtesting, and a full-featured screener inspired by screener.in × groww.in — all wrapped in two production-ready Streamlit apps.**

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Project Architecture](#project-architecture)
3. [SENTRAL — Complete Stock Analysis](#sentral--complete-stock-analysis)
4. [SCREENER — Fundamental + Technical Deep Dive](#screener--fundamental--technical-deep-dive)
5. [Streamlit Apps](#streamlit-apps)
6. [LLM-Powered Peer Discovery](#llm-powered-peer-discovery)
7. [Reddit Scraping — OAuth vs Anonymous](#reddit-scraping--oauth-vs-anonymous)
8. [API Keys Reference](#api-keys-reference)
9. [Quick Start](#quick-start)
10. [Output Files](#output-files)
11. [Technology Stack](#technology-stack)

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
| `SETUP.md` | Step-by-step setup guide |

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

SENTRAL is an 8-pillar multi-spectrum analysis pipeline:

### Pillar 1 — Fundamental Analysis
- **38 metrics**: P/E, P/B, P/S, EV/EBITDA, PEG, ROE, ROA, ROCE, Gross/Operating/Net margins, D/E, Current Ratio, Quick Ratio, Interest Coverage
- **Altman Z-Score** (5-factor bankruptcy predictor): Safe > 2.99, Grey 1.81–2.99, Distress < 1.81
- **Piotroski F-Score** (9-point profitability + leverage + efficiency): Strong ≥ 7, Moderate 4–6, Weak < 4
- **DCF Valuation**: 10-year discounted cash flow with configurable WACC and terminal growth
- Composite fundamental score normalised to [−1, +1]

### Pillar 2 — Technical Analysis (35 indicators)
| Category | Indicators |
|----------|------------|
| Moving Averages | SMA 10/20/50/100/200, EMA 10/20/50/200 |
| Momentum | RSI-14, Stochastic K/D, Williams %R, ROC-10 |
| Trend | MACD, MACD Signal, MACD Hist, ADX, +DI, −DI, CCI, DPO, Ichimoku A/B |
| Volatility | Bollinger Bands (±2σ), ATR, Keltner Channels, Ulcer Index |
| Volume | OBV, CMF, MFI, VWAP |
| Returns | 1d/5d/20d returns |

Risk metrics: Sharpe, Sortino, VaR (95%), CVaR, Max Drawdown, Calmar Ratio

### Pillar 3 — News Aggregation (13 sources)
| Source | Type |
|--------|------|
| Alpha Vantage | REST API |
| Finnhub | REST API |
| Tavily | AI search API |
| NewsAPI | REST API |
| EODHD | REST API |
| Marketaux | REST API |
| APILayer | REST API |
| Yahoo Finance RSS | RSS feed |
| Google News RSS | RSS feed |
| ET Markets RSS | RSS feed |
| Livemint RSS | RSS feed |
| Reddit RSS | Public RSS (no auth) |
| StockTwits | Public RSS |

### Pillar 4 — Sentiment Analysis (10 models)
| Model | Type |
|-------|------|
| NLTK VADER | Rule-based |
| FinBERT | Finance-tuned BERT |
| FinBERT-Tone | yiyanghkust/finbert-tone (w/ cardiffnlp fallback) |
| DistilRoBERTa | Lightweight RoBERTa |
| RoBERTa-Large | Large sentiment model |
| StockTwits-RoBERTa | Market sentiment specialist |
| Groq (6 models) | LLaMA 3.3 70B, LLaMA 4 Scout, Qwen3-32B, GPT-OSS |
| Gemini (3 models) | gemini-2.5-flash, lite, pro |

Ensemble score = weighted average across all models → label: Bullish / Neutral / Bearish

### Pillar 5 — ML Price Forecasting

#### LSTM Model Architecture
```
Input: (batch, seq_len=60, features=1)  ← rolling 60-day close price window
    ↓
LSTM Layer 1: hidden_size=128, dropout=0.2
    ↓
LSTM Layer 2: hidden_size=128, dropout=0.2
    ↓
Fully Connected: 128 → 64 → 1
    ↓
Output: predicted next-day price
```
- **Training**: 80/20 train-test split on historical close prices
- **Normalisation**: MinMaxScaler [0, 1] → inverse-transformed for output
- **Loss**: MSE; **Optimizer**: Adam, lr=0.001, epochs=50
- **Forecast**: Rolls window forward 30 days to produce daily price path

#### Temporal Transformer Architecture
```
Input: (batch, seq_len=60, d_model=64)  ← projected by Linear(1, 64)
    ↓
Positional Encoding: sinusoidal, max_len=500
    ↓
TransformerEncoder (2 layers):
    - nhead=8 attention heads
    - dim_feedforward=256
    - dropout=0.1
    ↓
Global Average Pooling (mean across seq_len)
    ↓
Fully Connected: 64 → 1
    ↓
Output: predicted next-day price
```
- Trained in parallel with LSTM using same dataset split
- Captures long-range dependencies more effectively than LSTM for liquid stocks

#### Monte Carlo GBM
```
dS = μ·S·dt + σ·S·dW
```
- **μ** (drift): annualised from historical log returns
- **σ** (volatility): annualised standard deviation of log returns
- **Simulations**: 1,000–5,000 paths (configurable in sidebar)
- **Horizon**: 30 trading days forward
- **Output**: mean path ± 1σ band, P(profit > 0) metric

#### Forecast Ensemble
```
Ensemble Price = 0.5 × LSTM_pred + 0.5 × Transformer_pred
Forecast Score = (Ensemble_30d_return − threshold) / normalisation_factor
```
The forecast score is clipped to [−1, +1] for signal integration.

---

### Pillar 6 — Signal Engine

#### Full Score Computation
```
Component Scores (each normalised to [−1, +1]):
  F_score  = fundamental_score(38 metrics)
  T_score  = technical_score(35 indicators → directional consensus)
  S_score  = sentiment_score(ensemble of 10 models)
  ML_score = forecast_score(LSTM + Transformer 30-day return)

Final Signal Score:
  S = 0.30 × F_score + 0.30 × T_score + 0.20 × S_score + 0.20 × ML_score

Decision:
  S > +0.15  → BUY  🟢  (confidence = S × 100%)
  S < −0.15  → SELL 🔴  (confidence = |S| × 100%)
  otherwise  → HOLD 🟡
```

#### Fundamental Sub-score (detail)
Each metric is normalised against sector medians and clipped to [−1, +1]:
- **Valuation**: low P/E, P/B, EV/EBITDA → positive; high multiples → negative
- **Profitability**: high ROE, ROA, Net Margin, Operating Margin → positive
- **Growth**: revenue growth, earnings growth → positive if > 15%
- **Safety**: low D/E, high Current Ratio, positive FCF → positive
- **Health**: Piotroski ≥ 7 → +0.3; Altman Z ≥ 3 → +0.2

#### Technical Sub-score (detail)
Each indicator votes +1 (bullish) or −1 (bearish) based on standard thresholds:
- **RSI**: < 30 → bullish, > 70 → bearish
- **MACD**: line > signal → bullish
- **Bollinger**: price below lower band → bullish
- **ADX**: > 25 with +DI > −DI → bullish
- **SMA/EMA crossovers**: short above long → bullish
- Aggregate vote / number of indicators → score in [−1, +1]

#### Sentiment Sub-score (detail)
```
Raw scores per model: {bullish: +1, neutral: 0, bearish: −1}

Model weights:
  - FinBERT (finance-tuned)   : 1.5×
  - FinBERT-Tone              : 1.5×
  - LLM (Groq/Gemini)         : 2.0×  (if available)
  - RoBERTa-based models      : 1.2×
  - VADER (rule-based)        : 0.8×  (lower weight)

S_score = weighted_avg(scores) clipped to [−1, +1]
```

---

### Pillar 7 — 20-Strategy Backtesting

All strategies are backtested over the full historical period (up to 5 years) with realistic assumptions:
- **Commission**: 0.1% per trade round-trip
- **Position sizing**: 100% equity per signal
- **Execution**: end-of-day signal, next-bar execution

| Category | Strategies |
|----------|------------|
| Trend-Following | EMA+MACD+RSI, Golden/Death Cross (SMA 50/200), Triple EMA (9/21/55) |
| Momentum | MACD crossover, ROC, Williams %R |
| Mean-Reversion | RSI Oversold/Overbought, Bollinger Bands, Stochastic |
| Volume | CMF Positive Flow, OBV Momentum, VWAP cross, MFI |
| Volatility | Keltner Channel Breakout, Donchian Channel |
| Multi-Factor | Multi-Confluence (5-indicator), Ichimoku Cloud, ADX Trend |
| Benchmark | Buy & Hold |

**Output per strategy** (in `backtest.csv`):
- Total return %, CAGR %, Sharpe, Sortino, Max Drawdown
- Win Rate %, Profit Factor, number of trades
- Kelly Criterion, suggested stop-loss

### Pillar 8 — AI Investment Thesis

Uses Gemini with a cascading fallback model chain:
```
gemini-2.5-flash → gemini-2.5-flash-lite → gemini-2.0-flash
```

The thesis prompt includes the following structured context:
- Current price, P/E, P/B, ROE, D/E, Altman Z, Piotroski score
- Composite signal (BUY/HOLD/SELL) and confidence %
- Technical indicator consensus (RSI, MACD, trend direction)
- Sentiment score and source count
- ML forecast return and direction
- Best-performing backtest strategy

**Output sections in the generated thesis:**
1. **Company Summary** — business description, competitive moat
2. **Bull Case** — key reasons to invest (data-driven)
3. **Bear Case** — key risks and concerns
4. **Valuation Assessment** — DCF vs Graham vs current price
5. **Technical Outlook** — trend, momentum, support/resistance
6. **Sentiment Context** — news tone, social media signals
7. **Recommendation** — final BUY/HOLD/SELL with confidence and time horizon

The thesis is rendered as formatted HTML in the app and included in the PDF report.

---

## SENTRAL App — Tab-by-Tab Guide

```bash
cd sentral_app && streamlit run app.py
```

| Tab | Contents |
|-----|----------|
| **📊 Signal Engine** | Composite score gauge, BUY/HOLD/SELL banner, pillar score breakdown (bar chart), confidence %, signal report download |
| **🔢 Fundamentals** | 38 metrics (4 tables: profitability/valuation/solvency/efficiency), DCF valuation, Graham Number, Altman Z gauge, Piotroski F scorecard, P&L + Balance Sheet + Cash Flow charts |
| **📈 Technical** | Interactive candlestick + 4 MA overlay, RSI + MACD + Stochastic subplots, Bollinger Bands, Ichimoku Cloud, ATR / OBV / CMF / VWAP charts, Risk metrics table (Sharpe, VaR, Max DD) |
| **📰 News & Sentiment** | News article feed (headlines + source + date), per-model sentiment scores, ensemble gauge, bullish/bearish word cloud, news count timeline |
| **🤖 ML Forecast** | LSTM vs Transformer vs ensemble 30-day forecast chart, Monte Carlo simulation fan (P(profit)), training loss curves, forecast confidence interval |
| **⚙️ Backtest** | All 20 strategy results table (sortable), equity curves for top 5 strategies, trade log for best strategy, drawdown comparison chart |
| **👥 Peers** | LLM-discovered peer list (with discovery method badge), peer comparison table (P/E, ROE, Net Margin, EV/EBITDA), scatter bubble chart (ROE vs P/E, size = market cap) |
| **🧠 AI Thesis** | Full AI-generated investment thesis (8 sections), export as PDF button, signal summary card |

**Sidebar controls:**
- Ticker (NSE/BSE/US stocks), Period (6m – 5y)
- Forecast horizon (7–120 days), LSTM look-back window (20–120 days)
- Monte Carlo simulations (1K–5K)
- Enable/disable ML (for faster load on slow connections)
- 10 API key text fields (saved to session state, never to disk)

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

A full replica of screener.in × groww.in built in Python. See [SCREENER.md](SCREENER.md) for detailed documentation.

**18 sections in the notebook:**
1. Company Overview (header metrics)
2. Price Chart (candlestick + MA)
3. Quarterly Results (12 quarters)
4. Annual P&L
5. Balance Sheet
6. Cash Flow Statement
7. Financial Ratios (4 categories, 25+ ratios)
8. Shareholding Pattern
9. Peer Comparison (LLM-powered)
10. Technical Analysis (RSI, MACD, BB, Stochastic)
11. Dividend History
12. Valuation Models (DCF, Graham Number, P/E reversion, Lynch PEG)
13. Multi-Stock Screener (custom query)
14. Corporate Events
15. OBV + Volume Analysis
16. Investment Calculator (SIP + Lump-sum)
17. Financial Health (Altman Z + Piotroski F)
18. Company Profile

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

**Sidebar controls:** ticker, period, Groq API key, Gemini API key

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
