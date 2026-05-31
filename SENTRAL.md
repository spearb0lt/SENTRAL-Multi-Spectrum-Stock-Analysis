# SENTRAL — Multi-Spectrum Stock Analysis Pipeline

> **SENTRAL (Sentiment + Technical + Fundamental Analysis) is an 8-pillar end-to-end stock analysis system combining fundamental scoring, 35-indicator technical analysis, 13-source news aggregation, 10-model sentiment ensemble, LSTM + Transformer price forecasting, 20-strategy backtesting, composite signal generation, and AI investment thesis — available as both a 82-cell Jupyter notebook and an 8-tab Streamlit app.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SENTRAL_Complete.ipynb — Notebook Guide](#sentral_completeipynb--notebook-guide)
3. [sentral_app — Streamlit App](#sentral_app--streamlit-app)
4. [Module Deep-Dives](#module-deep-dives)
5. [Signal Engine](#signal-engine)
6. [ML Forecasting](#ml-forecasting)
7. [Backtesting Engine](#backtesting-engine)
8. [Reddit OAuth Upgrade](#reddit-oauth-upgrade)
9. [Outputs Reference](#outputs-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     SENTRAL Pipeline                         │
│                                                              │
│  Ticker Input                                                │
│       │                                                      │
│       ├──→ [1] Data Loader   (yfinance + OHLCV + TA)        │
│       ├──→ [2] Fundamental   (38 metrics, Altman, Piotroski) │
│       ├──→ [3] Technical     (35 indicators, risk metrics)   │
│       ├──→ [4] News          (13 sources, 10 APIs + RSS)     │
│       ├──→ [5] Sentiment     (10 models, ensemble score)     │
│       ├──→ [6] Forecast      (LSTM + Transformer + MCM)      │
│       │                                                      │
│       ├──→ [7] Signal Engine                                 │
│       │         └── 0.30×Fundamental + 0.30×Technical        │
│       │             + 0.20×Sentiment + 0.20×Forecast         │
│       │             → BUY / HOLD / SELL + Confidence         │
│       │                                                      │
│       ├──→ [8] Backtest      (20 strategies)                 │
│       └──→ [9] Report        (HTML + PDF + JSON)             │
└─────────────────────────────────────────────────────────────┘
```

**Weight rationale:**
- Fundamental (0.30): Long-term business quality is the foundation of intrinsic value
- Technical (0.30): Price action reflects market consensus and short-term momentum
- Sentiment (0.20): News/social sentiment drives near-term volatility and narrative
- Forecast (0.20): ML prediction provides forward-looking context, but carries uncertainty

---

## SENTRAL_Complete.ipynb — Notebook Guide

The notebook contains **82 cells** organised into named sections. Run with kernel: *Python (SENTRAL venv)*.

### Setup Section (Cells 1–5)
- **Cell 1**: Library imports (torch, transformers, ta, yfinance, plotly, reportlab, etc.)
- **Cell 2**: Configuration block:
  ```python
  TICKER           = "HAL.NS"    # Any Yahoo Finance ticker
  PERIOD           = "5y"        # Historical data period
  BENCH_TICKER     = "^NSEI"     # Benchmark for comparison
  FORECAST_DAYS    = 30          # Days to forecast
  LOOKBACK         = 60          # LSTM sequence length
  N_SIMULATIONS    = 1000        # Monte Carlo paths
  USE_ML           = True        # Enable ML forecasting
  ```
- **Cell 3–4**: API key loading from `.env` (10 keys)
- **Cell 5**: Directory setup — creates `outputs/{TICKER}_{YYYYMMDD}_{HHMM}/`

### Pillar 1 — Data & Fundamentals (Cells 6–18)
- **Cell 6**: `download_stock_data(ticker, period)` — downloads OHLCV, financial statements, company info
- **Cell 7**: `compute_features(data)` — adds all 35 technical indicators to price DataFrame
- **Cells 8–12**: Fundamental metrics display — P/E, P/B, P/S, EV/EBITDA, PEG, ROE, ROA, margins, debt ratios
- **Cell 13**: Altman Z-Score computation and interpretation
- **Cell 14**: Piotroski F-Score (9 criteria check with PASS/FAIL display)
- **Cell 15**: DCF valuation (10-year with configurable WACC)
- **Cell 16**: Graham Number + Lynch PEG
- **Cell 17**: Fundamental score calculation → normalised [−1, +1]
- **Cell 18**: Peer comparison table (LLM-discovered, see peers module)

### Pillar 2 — Technical Analysis (Cells 19–26)
- **Cell 19**: Full indicator summary table with current signals
- **Cells 20–21**: Price chart with MA overlays (mplfinance + Plotly)
- **Cell 22**: Momentum chart: RSI + Stochastic + Williams %R
- **Cell 23**: Trend chart: MACD + ADX + CCI
- **Cell 24**: Volatility chart: Bollinger Bands + ATR + Keltner Channels
- **Cell 25**: Volume chart: OBV + CMF + MFI + VWAP
- **Cell 26**: Risk metrics — Sharpe, Sortino, VaR(95%), CVaR, Max Drawdown, Calmar Ratio

### Pillar 3 — News Aggregation (Cells 27–32)
- **Cell 27**: `fetch_all_news(ticker, company_name)` — queries all 13 sources
- **Cell 28**: Relevance filtering — score based on ticker/company name frequency
- **Cells 29–30**: News display by source, publication timeline
- **Cell 31**: News volume chart (articles per day over time)
- **Cell 32**: Top 20 articles table with relevance scores

#### News Sources Detail
| Source | Method | Coverage |
|--------|--------|----------|
| Alpha Vantage | REST API (`ALPHA_VANTAGE_API_KEY`) | Global + sentiment labels |
| Finnhub | REST API (`FINNHUB_API_KEY`) | Global + company news |
| Tavily | AI search API (`TAVILY_API_KEY`) | Deep web search |
| NewsAPI | REST API (`NEWSAPI_KEY`) | 70,000+ global sources |
| EODHD | REST API (`EODHD_API_KEY`) | Financial + company filings |
| Marketaux | REST API (`MARKETAUX_API_KEY`) | Market sentiment focus |
| APILayer | REST API (`APILAYER_API_KEY`) | Financial headlines |
| Yahoo Finance | RSS feed | Real-time Yahoo Finance articles |
| Google News | RSS feed | Broad news coverage |
| ET Markets | RSS feed | India-focused |
| Livemint | RSS feed | India business news |
| Reddit | RSS feed (upgradeable to OAuth) | Retail investor sentiment |
| StockTwits | RSS feed | Stock-specific social sentiment |

### Pillar 4 — Sentiment Analysis (Cells 33–42)
- **Cell 33**: `SentimentAnalyzer` class initialised with all 10 models
- **Cells 34–36**: VADER, FinBERT, RoBERTa inference on news corpus
- **Cell 37**: Groq LLM sentiment (6 models in cascade: llama-3.3-70b → llama-4-scout → llama-3.1-8b → qwen-32b → deepseek-r1 → mistral-saba)
- **Cell 38**: Gemini sentiment (gemini-2.5-flash → gemini-2.5-flash-lite → gemini-2.0-flash)
- **Cell 39**: Ensemble score computation (inverse-variance weighting)
- **Cell 40**: Sentiment distribution plot
- **Cell 41**: Per-source sentiment breakdown
- **Cell 42**: `sentiment_summary.csv` saved

#### Sentiment Models
| Model | Architecture | Specialisation |
|-------|-------------|----------------|
| NLTK VADER | Rule-based lexicon | General text |
| FinBERT (ProsusAI) | BERT | Financial news |
| FinBERT-Tone (yiyanghkust) | BERT | Tone analysis |
| DistilRoBERTa-Sentiment | DistilRoBERTa | Fast general sentiment |
| RoBERTa-Large-Sentiment | RoBERTa | High-accuracy general |
| twitter-roberta (cardiffnlp) | RoBERTa | Short-form social text |
| Groq llama-3.3-70b-versatile | LLaMA 3.3 70B | Reasoning + context |
| Groq llama-4-scout | LLaMA 4 | Multi-modal reasoning |
| Groq qwen-32b-preview | Qwen 32B | Multi-lingual |
| Gemini 2.5 Flash | Gemini | Long-context analysis |

### Pillar 5 — ML Forecasting (Cells 43–54)

#### LSTM Architecture
```
Input (seq_len=60, features=1)  →  LSTM (hidden=128, layers=2)
→ Dropout(0.2)  →  Linear(128, 64)  →  ReLU  →  Linear(64, 1)
```
- Training: 80% / 20% split, Adam optimiser, MSE loss, 50 epochs
- Normalisation: MinMaxScaler [0, 1]
- Saves to `lstm_model.pt`

#### Transformer Architecture
```
Input → Embedding → Positional Encoding → 
TransformerEncoder (nhead=8, layers=2, dim_feedforward=256) → 
Linear(d_model, 1)
```
- Saves to `transformer_model.pt`

#### Monte Carlo GBM
- Uses historical `μ` (mean daily return) and `σ` (std daily return)
- Simulates N paths: `S(t+1) = S(t) × exp((μ − σ²/2)dt + σ√dt × ε)`
- Outputs: P(profit), P10/P50/P90 percentile paths, Value at Risk

- **Cell 43–46**: Data preparation, scaling, sequence creation
- **Cell 47–50**: LSTM training loop with loss curve
- **Cell 51**: Transformer training
- **Cell 52**: Ensemble: `0.5×LSTM + 0.5×Transformer`
- **Cell 53**: Monte Carlo simulation (1000 paths default)
- **Cell 54**: Forecast chart + `forecast_30d.csv` saved

### Pillar 6 — Signal Engine (Cell 55)
See [Signal Engine](#signal-engine) section below.

### Pillar 7 — Backtesting (Cells 56–68)
See [Backtesting Engine](#backtesting-engine) section below.

### Pillar 8 — Report Generation (Cells 69–82)
- **Cell 69–72**: Signal summary dashboard display
- **Cell 73**: `generate_html_report()` — full interactive HTML report
- **Cell 74**: `generate_pdf_report()` — PDF via reportlab
- **Cell 75**: `signal_report.json` dump (all scores, signals, metrics)
- **Cells 76–80**: AI Investment Thesis via Gemini (structured prompt with all signals)
- **Cells 81–82**: Output file summary

---

## sentral_app — Streamlit App

### Launch
```bash
cd sentral_app
..\venv\Scripts\streamlit run app.py    # Windows
# or
../venv/bin/streamlit run app.py        # Linux/Mac
```

### Sidebar Controls
| Control | Default | Description |
|---------|---------|-------------|
| Ticker | `HAL.NS` | Target stock |
| Period | `5y` | Historical data |
| Forecast Days | 30 | LSTM forecast horizon |
| LSTM Lookback | 60 | Sequence window |
| Monte Carlo N | 1000 | Simulation paths |
| ML Forecast | ON | Toggle LSTM + Transformer |
| 10 API key fields | — | Loaded from `.env` |
| Upload Bundle ZIP | — | Restore a previously saved analysis bundle |
| Load from Bundle | — | Skip all heavy processing, display instantly |

### Download / Upload Bundle
After any successful analysis a four-button row appears **above the tabs**:

| Button | File | Contents |
|--------|------|----------|
| ⬇ HTML Report | `SENTRAL_{TICKER}_{DATE}.html` | Full interactive report |
| ⬇ PDF Report | `SENTRAL_{TICKER}_{DATE}.pdf` | Static PDF (requires `reportlab`) |
| ⬇ Price + Indicators CSV | `{TICKER}_indicators.csv` | OHLCV + 35 TA indicator columns |
| 📦 Download Full Bundle ZIP | `SENTRAL_{TICKER}_{DATE}_bundle.zip` | Complete analysis bundle (see table below) |

**Bundle ZIP contents:**

| File | Contents |
|------|----------|
| `price_indicators.csv` | OHLCV + all 35 TA columns, DatetimeIndex |
| `news_corpus.csv` | All fetched articles |
| `sentiment_analysis.csv` | Per-model scores + ensemble |
| `peers.csv` | Peer comparison table |
| `backtest_results.csv` | 20-strategy metrics (raw floats) |
| `backtest_cumuls.csv` | Cumulative return curves per strategy |
| `ml_forecast.csv` | 30-day LSTM / Transformer / Ensemble predictions |
| `ml_eval.csv` | RMSE + MAPE for both models |
| `session_data.json` | All scalars: signal, Altman Z, Piotroski, DCF, Kelly, thesis, etc. |

**To restore from a bundle:**
1. In the sidebar, upload the ZIP using **Upload Bundle ZIP**.
2. Click **⚡ Load from Bundle**.
3. All 8 tabs populate instantly with no API calls.

What is **re-computed** on load (pure numpy/pandas, ~2 s total):
`compute_risk_metrics`, `detect_candlestick_patterns`, `compute_seasonality`,
`compute_technical_score`, `get_risk_flags`, `run_monte_carlo`

What is **skipped** (loaded directly from ZIP):
- News fetch (13 API sources)
- Sentiment inference (5 HuggingFace models + Groq + Gemini)
- LSTM + Transformer training
- 20-strategy backtest

### Tab 0 — Signal Engine
- Large **BUY/HOLD/SELL** badge with signal color
- Confidence percentage (0–100%)
- 4 pillar score bars: Fundamental, Technical, Sentiment, Forecast
- Risk flags (up to 5 warnings: high D/E, extreme RSI, etc.)

### Tab 1 — Fundamentals
- 38-metric display across 5 categories
- DCF intrinsic value card
- Graham Number card
- Altman Z-Score + Piotroski F-Score
- Annual P&L + Balance Sheet tables

### Tab 2 — Technical
- 6-row interactive chart: Price+MA, RSI, MACD, Stochastic, BB, Volume
- Risk metrics: Sharpe, Sortino, VaR, CVaR, Max Drawdown
- Current indicator signals table

### Tab 3 — News & Sentiment
- Article feed from all sources
- Sentiment distribution pie chart
- Per-source sentiment bars
- Model agreement chart

### Tab 4 — ML Forecast
- 30-day LSTM + Transformer ensemble chart
- Monte Carlo percentile fan chart (P10/P50/P90)
- Probability of profit metric
- Forecast vs actual backtesting chart

### Tab 5 — Backtest
- 20-strategy results table
- Best strategy highlight
- Cumulative return chart (top 5 strategies)
- Strategy parameter details expandable

### Tab 6 — Peers
- LLM discovery badge + method
- 12-column comparison table
- P/E comparison bar, Market Cap bar, ROE vs P/B scatter

### Tab 7 — AI Thesis
- Full Gemini-generated investment thesis
- Key points: Bull case, Bear case, Catalysts, Risks
- Data sources cited

---

## Module Deep-Dives

### `modules/data_loader.py`

```python
download_stock_data(ticker: str, period: str) -> dict
```
Returns pickle-safe dict with: `{info, data, hist, fin, bal, cf, company_name, sector, industry, currency, currency_sym, exchange}`  
**Important:** Does NOT include `yfinance.Ticker` object (not pickle-serializable).

```python
compute_features(data: pd.DataFrame) -> pd.DataFrame
```
35 indicators computed using `ta` library (v0.11.0):
- `ta.volume.VolumeWeightedAveragePrice` (note: NOT `VolumeWeightedAveragePriceIndicator`)

### `modules/fundamental.py`
```python
compute_fundamental_score(info, bal, fin, cf) -> float  # [-1, +1]
compute_altman_z(bal, fin) -> tuple[float, str]
compute_piotroski(bal, fin, cf) -> tuple[int, dict]
compute_dcf(cf, info, wacc, growth_rate) -> dict
```

### `modules/technical.py`
```python
compute_technical_score(data) -> float          # [-1, +1]
compute_risk_metrics(data) -> dict              # Sharpe, Sortino, VaR, etc.
detect_patterns(data) -> list[str]             # Head & Shoulders, etc.
```

### `modules/news.py`
```python
fetch_all_news(ticker, company_name, keys: dict) -> list[dict]
    # Each article: {title, body, source, published, url, relevance_score}
```

### `modules/sentiment.py`
```python
class SentimentAnalyzer:
    def analyze_batch(articles: list[dict]) -> dict  # model → score mapping
    def ensemble_score(scores: dict) -> tuple[float, str]  # (score, label)
```

### `modules/ml_forecast.py`
```python
class LSTMModel(nn.Module): ...
class TransformerModel(nn.Module): ...

def train_lstm(data, lookback, hidden_size, num_layers, epochs) -> tuple[model, scaler]
def train_transformer(data, lookback, d_model, nhead, epochs) -> tuple[model, scaler]
def ensemble_forecast(lstm, transformer, data, days) -> pd.DataFrame
def monte_carlo_simulation(data, days, n_sims) -> dict
```

### `modules/signals.py`
```python
def compute_composite_signal(fundamental_score, technical_score,
                               sentiment_score, forecast_score,
                               weights=(0.30, 0.30, 0.20, 0.20)) -> dict
    # Returns: {signal, composite_score, confidence, risk_flags}
```

### `modules/backtest.py`
```python
def run_all_strategies(data: pd.DataFrame) -> pd.DataFrame
    # 20 strategies × (return, CAGR, Sharpe, max_drawdown, win_rate, trades)
def get_optimal_strategy(results: pd.DataFrame) -> str
```

### `modules/peers.py`
```python
def get_peer_data(ticker, sector, company_name, exchange, groq_key, gemini_key)
    -> tuple[list, pd.DataFrame, str]   # (tickers, df_peers, method)
```
3-tier discovery: Groq → Gemini → hard-coded `SECTOR_PEERS` dict.

### `modules/report.py`
```python
def generate_html_report(ticker, all_data: dict, output_dir: str) -> str  # path
def generate_pdf_report(ticker, all_data: dict, output_dir: str) -> str   # path
```

---

## Signal Engine

The composite signal combines all four pillars:

```
Composite = w₁·F + w₂·T + w₃·S + w₄·M

Where:
  F = Fundamental score  ∈ [-1, +1]   weight w₁ = 0.30
  T = Technical score    ∈ [-1, +1]   weight w₂ = 0.30
  S = Sentiment score    ∈ [-1, +1]   weight w₃ = 0.20
  M = ML Forecast score  ∈ [-1, +1]   weight w₄ = 0.20
```

**Signal thresholds:**
| Score | Signal | Meaning |
|-------|--------|---------|
| > +0.15 | 🟢 BUY | Strong positive alignment across pillars |
| −0.15 to +0.15 | 🟡 HOLD | Mixed or neutral signals |
| < −0.15 | 🔴 SELL | Strong negative alignment |

**Confidence formula:**
```
Confidence = min(100, (|Composite| / 1.0) × 100 × (1 + pillar_agreement))
```
Where `pillar_agreement` ∈ [0, 1] measures how many pillars agree on direction.

**Risk flags triggered when:**
- Altman Z-Score < 1.81 (distress zone)
- D/E ratio > 3.0
- RSI > 80 (extremely overbought)
- RSI < 20 (extremely oversold)
- Max drawdown > 40%
- VaR(95%) daily loss > 5%

---

## Backtesting Engine

20 strategies implemented as vectorised signals on historical OHLCV data:

| # | Strategy | Signal Condition |
|---|----------|-----------------|
| 1 | EMA+MACD+RSI | EMA cross AND MACD bullish AND RSI 30–70 |
| 2 | Golden/Death Cross | SMA50 crosses SMA200 |
| 3 | MACD Signal Cross | MACD crosses Signal line |
| 4 | RSI Oversold Bounce | RSI crosses up through 30 |
| 5 | RSI Overbought Exit | RSI crosses down through 70 |
| 6 | Bollinger Band Mean Rev | Price touches lower/upper band |
| 7 | ADX Trend Following | ADX > 25 AND +DI > -DI |
| 8 | Stochastic Crossover | %K crosses %D in oversold/overbought zone |
| 9 | Williams %R | %R crosses −80 (buy) / −20 (sell) |
| 10 | CMF Positive Flow | CMF crosses 0 from below |
| 11 | OBV Momentum | OBV > OBV_SMA20 |
| 12 | VWAP Cross | Price crosses above/below VWAP |
| 13 | MFI Signal | MFI crosses 20 (buy) / 80 (sell) |
| 14 | ROC Signal | ROC crosses 0 from below/above |
| 15 | Triple EMA | EMA10 > EMA20 > EMA50 |
| 16 | Donchian Channel | Price breaks 20-day high/low |
| 17 | Keltner Breakout | Price breaks above/below Keltner channels |
| 18 | Multi-Confluence | ≥5 indicators agree on direction |
| 19 | Ichimoku Cloud | Price above/below cloud + Tenkan/Kijun cross |
| 20 | Buy & Hold | Always long (baseline) |

**Backtest metrics computed:**
- Total Return %
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (risk-free rate = 6%)
- Max Drawdown %
- Win Rate % (profitable trades / total trades)
- Number of Trades
- Kelly Criterion (optimal position size)

---

## Reddit OAuth Upgrade

Currently SENTRAL uses public Reddit RSS feeds:
```python
# sentral_app/modules/news.py
url = f"https://www.reddit.com/search.rss?q={company}&limit=25"
```

This works but is rate-limited and shallow. Using **PRAW (Python Reddit API Wrapper)** with OAuth unlocks:

| Feature | RSS (current) | OAuth (PRAW) |
|---------|:---:|:---:|
| Rate limit | ~60/hour | 60/minute |
| Post body text | ❌ | ✅ |
| Comment text | ❌ | ✅ |
| Upvote counts | ❌ | ✅ |
| Subreddit-specific search | ❌ | ✅ |
| Time filtering | ❌ | ✅ (week/month/year) |
| Upvote-weighted sentiment | ❌ | ✅ |
| Access to moderated subs | ❌ | ✅ (if authorised) |

**Recommended subreddits for Indian stocks:**
- `r/IndianStockMarket` — NSE/BSE stock discussions
- `r/IndiaInvestments` — Long-term investment discussions
- `r/SecurityAnalysis` — Deep fundamental analysis
- `r/stocks` — Global stock discussion
- `r/investing` — Portfolio and strategy

**Setup:**
1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Create app → type: **script** → redirect URI: `http://localhost:8080`
3. Note `client_id` (under app name) and `client_secret`
4. Add to `.env`:
   ```env
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USERNAME=your_reddit_username
   ```
5. Install PRAW: `pip install praw`
6. Enable in `news.py` by replacing the RSS fetch with the PRAW implementation (commented out in the module, ready to enable)

---

## Outputs Reference

Each run creates `outputs/{TICKER}_{YYYYMMDD}_{HHMM}/`:

### `signal_report.json`
```json
{
  "ticker": "HAL.NS",
  "company_name": "Hindustan Aeronautics Limited",
  "timestamp": "2025-05-31T00:16:42",
  "signal": "BUY",
  "composite_score": 0.287,
  "confidence": 72.4,
  "pillar_scores": {
    "fundamental": 0.41,
    "technical": 0.23,
    "sentiment": 0.18,
    "forecast": 0.31
  },
  "risk_flags": [],
  "fundamental_metrics": { ... },
  "technical_metrics": { ... },
  "backtest_best_strategy": "Multi-Confluence",
  "forecast_p50_30d": 4823.5
}
```

### `price_indicators.csv`
`Date, Open, High, Low, Close, Volume, SMA20, SMA50, SMA200, EMA20, RSI, MACD, ...` (40+ columns)

### `forecast_30d.csv`
`Date, LSTM, Transformer, Ensemble, MC_P10, MC_P50, MC_P90`

### `backtest.csv`
`Strategy, Total_Return, CAGR, Sharpe, Max_Drawdown, Win_Rate, Trades, Kelly`

### `news_corpus.csv`
`Title, Body, Source, Published, URL, Relevance_Score`

### `sentiment_summary.csv`
`Model, Score, Label, Confidence, Article_Count`

### `lstm_model.pt` / `transformer_model.pt`
PyTorch state dicts — loadable with:
```python
model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load("lstm_model.pt"))
```

---

*SENTRAL — Multi-Spectrum Stock Analysis  ·  Not financial advice  ·  For research and education only*
