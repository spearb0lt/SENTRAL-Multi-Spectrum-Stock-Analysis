# SCREENER — Deep Fundamental & Technical Stock Analysis

> **A Python implementation of screener.in × groww.in — covering company overview, quarterly results, annual financials, balance sheet, cash flow, 25+ financial ratios, shareholding patterns, LLM-powered peer discovery, interactive charts, valuation models, technical indicators, multi-stock screener, investment calculator, and financial health scores.**

---

## Table of Contents

1. [Overview](#overview)
2. [screener.ipynb — Notebook Sections](#screeneripynb--notebook-sections)
3. [screener_app — Streamlit App](#screener_app--streamlit-app)
4. [Module Reference](#module-reference)
5. [LLM Peer Discovery](#llm-peer-discovery)
6. [Multi-Stock Screener Engine](#multi-stock-screener-engine)
7. [Valuation Models](#valuation-models)
8. [Configuration & Usage](#configuration--usage)

---

## Overview

SCREENER replicates the functionality of two of India's most popular financial platforms:

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

**Data source:** Yahoo Finance via `yfinance` (real-time delayed).  
**Supported tickers:** NSE (`.NS`), BSE (`.BO`), US markets (`AAPL`), and most Yahoo Finance listed stocks globally.

---

## screener.ipynb — Notebook Sections

Run the notebook cell-by-cell in VS Code (kernel: *Python (SENTRAL venv)*) or JupyterLab.

### Cell 1 — Imports
All required libraries: `numpy`, `pandas`, `yfinance`, `ta`, `pandas_ta`, `plotly`, `mplfinance`, `scipy`, `IPython.display`. Sets matplotlib dark style.

### Cell 2 — Configuration
```python
TICKER  = "HAL.NS"      # Change to any Yahoo Finance symbol
PERIOD  = "5y"          # 1y / 2y / 3y / 5y / 10y / max
BENCH   = "^NSEI"       # ^NSEI (Nifty), ^GSPC (S&P500), ^BSE200
```
Loads `GROQ_API_KEY` and `GEMINI_API_KEY` from `.env` for peer discovery.

### Cell 3 — Data Download
Downloads all data for the configured ticker:
- `hist` — OHLCV price history
- `fin`, `bal`, `cf` — Annual financials (income statement, balance sheet, cash flow)
- `qfin`, `qbal`, `qcf` — Quarterly financials
- `actions` — Dividends and splits
- `holders_inst`, `holders_major` — Institutional and major holders
- `calendar` — Earnings dates, dividend dates
- `bench_hist` — Benchmark price history for relative returns

### Section 1 — Company Overview (Cells 4–6)
**Cell 5** renders a HTML card showing:
- Current price, change %, 52-week high/low, market cap
- P/E (trailing + forward), P/B, EV/EBITDA, PEG ratio
- ROE, ROA, ROCE, net margin, gross margin
- D/E ratio, interest coverage, current ratio, quick ratio
- Dividend yield, EPS (TTM), book value per share
- Exchange, sector, industry

**Cell 6** auto-generates **Pros & Cons** analysis:
- 12+ criteria checked: valuation, profitability, solvency, growth, dividends, liquidity
- Color-coded: 🟢 Strengths / 🔴 Concerns

### Section 2 — Price Chart (Cells 7–9)
**Cell 8:** Interactive Plotly candlestick chart with:
- SMA20, SMA50, SMA200 overlays
- EMA20 overlay
- Volume bars (green/red)
- Date range slider

**Cell 9:** Period returns vs benchmark:
- Bar chart showing stock vs benchmark returns for 1M, 3M, 6M, 1Y, 3Y periods

### Section 3 — Quarterly Results (Cells 10–11)
**Cell 11:** 12-quarter table with:
- Revenue, Operating Income, Net Profit (in ₹ Cr)
- Operating Margin %, Net Margin %, EPS
- Colour-coded margin gradient
- 4-subplot charts: Revenue, Net Profit, OPM%, EPS

### Section 4 — Annual P&L (Cells 12–13)
**Cell 13:** Annual P&L table + charts:
- Revenue, Operating Income, Net Profit, EPS
- Gross/Operating/Net margin trend lines
- Revenue + profit grouped bar chart

### Section 5 — Balance Sheet (Cells 14–15)
**Cell 15:** Annual balance sheet table + charts:
- Total Assets, Total Liabilities, Stockholders' Equity
- Debt/Equity ratio, Quick Ratio trends
- Grouped bar chart: Assets vs Liabilities vs Equity

### Section 6 — Cash Flow (Cells 16–17)
**Cell 17:** Annual cash flow table + grouped bar chart:
- Operating Cash Flow, Investing Cash Flow, Financing Cash Flow
- Free Cash Flow (Operating CF − CapEx)

### Section 7 — Financial Ratios (Cells 18–20)
**Cell 19:** Four ratio categories displayed as styled DataFrames:

| Category | Ratios |
|----------|--------|
| **Valuation** | P/E (trailing + forward), P/B, P/S, EV/EBITDA, EV/Revenue, PEG |
| **Profitability** | ROE, ROA, Gross Margin, Operating Margin, Net Margin, EBITDA Margin |
| **Solvency** | D/E, Current Ratio, Quick Ratio, Interest Coverage, Debt/EBITDA |
| **Per Share** | EPS (TTM + diluted), Book Value/share, Revenue/share, FCF/share |

**Cell 20:** **Fundamental Scorecard** — radar chart (6 dimensions, normalised 0–1):
- ROE, Net Margin, Gross Margin, Current Ratio, Value (inverse P/E), Safety (inverse D/E)

### Section 8 — Shareholding Pattern (Cells 21–22)
**Cell 22:** Donut pie chart + table:
- Promoter %, Institutional %, Public/Other %
- Top institutional holders table (15 rows)

### Section 9 — Peer Comparison (Cells 23–26)
**Cell 24:** LLM-powered peer discovery (see [LLM Peer Discovery](#llm-peer-discovery))

**Cell 25:** Peer metrics table with `highlight_max` / `highlight_min` styling:
- Name, Price, P/E, P/B, ROE%, ROCE%, Div Yield%, Mkt Cap (Cr), EV/EBITDA, Beta, 52W High, 52W Low

**Cell 26:** 4-chart grid + ROE vs P/E bubble scatter

### Section 10 — Technical Analysis (Cells 27–29)
**Cell 28:** Computes all indicators and shows a signals summary table:
- RSI (Overbought/Oversold), MACD (Bullish/Bearish), BB (Above/Below/Inside), ADX (Trending/Ranging)
- Stochastic, OBV, Ichimoku (above/below cloud)

**Cell 29:** 4-row interactive Plotly chart:
1. Price + Bollinger Bands (upper, middle, lower)
2. RSI-14 with 30/70 reference lines
3. MACD histogram + signal line
4. Stochastic %K/%D with 20/80 levels

### Section 11 — Dividend History (Cells 30–31)
**Cell 31:** Dividend charts:
- Per-event bar chart (all historical dividends)
- Annual totals bar chart
- Summary stats: total paid, average annual yield, years of data

### Section 12 — Valuation Models (Cells 32–33)
**Cell 33:** Four models computed and compared (see [Valuation Models](#valuation-models))

### Section 13 — Multi-Stock Screener (Cells 34–35)
**Cell 35:** Configure `SCREEN_TICKERS` and `FILTERS`:
```python
SCREEN_TICKERS = ["HDFCBANK.NS", "INFY.NS", "TCS.NS", ...]  # 30+ default
FILTERS = {
    "pe_ratio":          ("<=", 30),
    "price_to_book":     ("<=", 5),
    "return_on_equity":  (">=", 0.15),
    "net_margin":        (">=", 0.10),
    "debt_to_equity":    ("<=", 1.0),
    "market_cap":        (">=", 5e10),
}
```

### Section 14 — Corporate Events (Cells 36–37)
**Cell 37:** Upcoming events and history:
- Next earnings date + estimate
- Ex-dividend date, payment date
- Splits history with split ratio

### Section 15 — OBV + Volume Indicators (Cells 38–39)
**Cell 39:** 3-row chart: OBV (On-Balance Volume), CMF (Chaikin Money Flow), MFI (Money Flow Index)

### Section 16 — Investment Calculator (Cells 40–41)
**Cell 41:** SIP + lump-sum calculator:
- Inputs: monthly SIP, lump-sum, horizon (years), expected CAGR %
- Shows: historical CAGR from `hist` data
- Output: SIP final value, lump-sum final value, growth chart

### Section 17 — Financial Health (Cells 42–43)
**Cell 43:**

**Piotroski F-Score (9 criteria):**
| # | Criterion | Pass |
|---|-----------|------|
| 1 | Positive ROA | ROA > 0 |
| 2 | Positive Operating Cash Flow | OCF > 0 |
| 3 | ROA improving | ROA YoY > 0 |
| 4 | Accruals low | OCF/Assets > ROA |
| 5 | Leverage decreasing | D/E YoY lower |
| 6 | Liquidity improving | Current ratio YoY higher |
| 7 | No dilution | Shares not increased |
| 8 | Gross margin improving | YoY > 0 |
| 9 | Asset turnover improving | Revenue/Assets YoY > 0 |

**Altman Z-Score:**
- Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(ME/TL) + 1.0×(Sales/TA)
- > 2.99 = Safe Zone, 1.81–2.99 = Grey Zone, < 1.81 = Distress Zone

### Section 18 — Company Profile (Cells 44–45)
**Cell 45:** HTML card with:
- Full company description (up to 1200 chars)
- HQ city + country, full-time employees
- Website link, CEO name, all listed officers

---

## screener_app — Streamlit App

### Launch
```bash
cd screener_app
..\venv\Scripts\streamlit run app.py    # Windows
# or
../venv/bin/streamlit run app.py        # Linux/Mac
```
Opens at `http://localhost:8501`

### Sidebar
| Control | Description |
|---------|-------------|
| Ticker | Any Yahoo Finance symbol (e.g. `HAL.NS`, `AAPL`, `RELIANCE.BO`) |
| Historical Period | 1y / 2y / 3y / 5y / 10y / max |
| Benchmark | Index to compare returns against |
| Groq API Key | For LLM peer discovery |
| Gemini API Key | Fallback LLM for peer discovery |
| Save keys to .env | Checkbox → Save button |
| Analyse Stock | Trigger data load + analysis |
| Upload Bundle ZIP | Restore a previously saved analysis bundle |
| Load from Bundle | Skip all downloading and restore from ZIP instantly |

### Download / Upload Bundle
After any successful analysis a two-button row appears **above the tabs**:

| Button | File | Contents |
|--------|------|----------|
| 📦 Download Analysis Bundle ZIP | `SCREENER_{TICKER}_{DATE}_bundle.zip` | `df_ta.csv`, `bench_hist.csv`, `fin.csv`, `bal.csv`, `cf.csv`, `qfin.csv`, `peers.csv`, `holders_inst.csv`, `session_data.json` |
| ⬇ Price + Indicators CSV | `{TICKER}_indicators.csv` | OHLCV + all technical indicators |

**To restore from a bundle:**
1. In the sidebar, upload the ZIP using **Upload Bundle ZIP**.
2. Click **⚡ Load from Bundle**.
3. All 11 tabs populate instantly — no yfinance calls, no peer re-fetching.

What is **re-computed** on load (all pure pandas/numpy, < 1 s):
- Technical indicators (`compute_technical_indicators`) are already embedded in `df_ta.csv` — no recomputation needed.

What is **skipped** (loaded directly from the ZIP):
- yfinance financial statement downloads (`fin`, `bal`, `cf`, `qfin`)
- Benchmark download (`bench_hist`)
- Peer discovery (LLM or sector-database)

### Tab 0 — Overview
- 12-metric grid: 52W High/Low, P/B, EPS, ROE, ROA, Div Yield, D/E, Beta, Fwd P/E, Book Value, Exchange
- **52-week range progress bar** — visual position between 52W low and high
- Pros & Cons (12+ criteria auto-generated)
- "About the Company" expander: description, HQ, employees, website

### Tab 1 — Price Chart
- **Chart type toggle**: Candlestick / Line / OHLC
- **MA selector**: SMA20, SMA50, SMA200, EMA20, EMA50
- **Range selector**: 1M, 3M, 6M, 1Y, 2Y, 5Y, All
- Volume bars (green/red)
- **Period returns vs benchmark** bar chart

### Tab 2 — Financials (3 sub-tabs)
- **P&L**: Annual table with margin gradients + Revenue/Profit bar + Margin trend line charts
- **Balance Sheet**: Annual table + Assets vs Liabilities vs Equity grouped bar
- **Cash Flow**: Annual table + OCF/ICF/FCF grouped bar

### Tab 3 — Ratios
- 4 ratio category tables (2×2 grid layout)
- Radar chart scorecard (6 normalised dimensions)
- Altman Z-Score with colour signal
- Piotroski F-Score with all 9 criteria in expander

### Tab 4 — Quarterly
- 12-quarter table (Revenue, Net Profit, OPM%, NPM%, EPS) with gradient
- 4-subplot chart: Revenue, Net Profit, OPM%, EPS

### Tab 5 — Shareholding
- Donut pie chart (Promoter / Institutional / Public)
- Top 15 institutional holders (expandable)

### Tab 6 — Peers
- Discovery method badge: `🤖 LLM-discovered` or `📚 Sector database`
- Peer comparison table (12 columns)
- 3 chart sub-tabs: P/E comparison, Market Cap, ROE vs P/B scatter bubble

### Tab 7 — Valuation
- **DCF**: WACC slider (8–20%), terminal growth slider (2–8%), intrinsic value + upside %
- **Graham Number**: √(22.5 × EPS × Book Value), upside %
- **P/E Reversion**: sector-median P/E × EPS, upside %
- Summary table: all models side-by-side

### Tab 8 — Technical
- RSI, MACD, ADX, ATR metric cards at top
- 4-row chart: Price+BB / RSI / MACD / Stochastic
- OBV + CMF chart below

### Tab 9 — Screener
- Filter panel: P/E ≤, P/B ≤, ROE% ≥, Net Margin% ≥, D/E ≤, Mkt Cap (Cr) ≥, Rev Growth% ≥, Div Yield% ≥
- Runs on **60 NSE stocks** across 10 sectors
- Sortable result table with coloured gradients
- **CSV download** button *(screener results only; not included in the bundle ZIP)*

### Tab 10 — Calculator
- SIP final value + lump-sum final value
- Historical CAGR from price data pre-filled
- Growth chart: SIP vs lump-sum vs invested capital over time

---

## Module Reference

### `modules/data_loader.py`

```python
download_full_data(ticker: str, period: str) -> dict
```
Returns: `{info, hist, fin, bal, cf, qfin, qbal, qcf, actions, holders_inst, holders_major, calendar, company_name, sector, industry, exchange, currency, currency_sym}`  
All values are pickle-safe (no yfinance.Ticker objects).

```python
download_benchmark(bench: str, period: str) -> pd.DataFrame
```

```python
compute_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame
```
Adds 25+ columns: SMA10/20/50/100/200, EMA20/50/200, RSI, Stoch K/D, ROC, Williams_R, MACD/Sig/Hist, ADX/Pos/Neg, BB_Upper/Middle/Lower/Pct, ATR, OBV, CMF, MFI, VWAP, Ichimoku A/B.

### `modules/fundamentals.py`

```python
build_pl_table(fin, info, cur)      -> pd.DataFrame  # Annual P&L
build_balance_sheet(bal, cur)       -> pd.DataFrame  # Annual balance sheet
build_cashflow(cf, cur)             -> pd.DataFrame  # Annual cash flows
compute_all_ratios(info)            -> dict           # 4 categories, 25+ ratios
compute_piotroski_fscore(bal, fin, cf, info) -> dict # score, signal, criteria dict
compute_altman_z(info)              -> dict           # z_score, zone
compute_dcf(cf, info, wacc, tg)     -> dict           # intrinsic, upside, g_rate (or error)
compute_graham_number(info)         -> float | None   # Graham Number
```

### `modules/peers.py`

```python
get_peer_data(ticker, sector, company_name, exchange, groq_key, gemini_key)
    -> tuple[list, pd.DataFrame, str]   # (peer_list, df_peers, method)
```
`df_peers` columns: `Ticker, Name, Price, P/E, P/B, ROE %, ROCE %, Div Yield %, Mkt Cap (Cr), EV/EBITDA, Beta, 52W High, 52W Low`

### `modules/screener.py`

```python
DEFAULT_UNIVERSE: list[str]   # 60 NSE tickers across 10 sectors

fetch_screener_data(tickers: tuple) -> pd.DataFrame
    # 16 metrics per stock, @st.cache_data(ttl=1800)

apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame
    # filters = {"column": ("op", value)}  op in [<=, >=, ==, <, >]
```

---

## LLM Peer Discovery

The peer discovery chain:

```
groq (llama-3.3-70b-versatile)
  ↓ fails or no key
groq (llama-3.1-8b-instant)
  ↓ fails or no key
gemini-2.5-flash
  ↓ fails or no key
gemini-2.5-flash-lite
  ↓ fails
SECTOR_PEERS hard-coded database (12 sectors)
```

**Hard-coded sectors**: Technology, Financials, Consumer Cyclical, Healthcare, Energy, Industrials, Consumer Defensive, Utilities, Communication Services, Materials, IT Services India, Banking India.

Each sector contains 8–12 well-known tickers as a reliable fallback.

---

## Multi-Stock Screener Engine

### Default Universe (60 stocks)
10 sectors represented: IT (TCS, INFY, WIPRO, HCL, TECHM), Banking (HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN), Auto (TATAMOTORS, M&M, MARUTI, BAJAJ-AUTO, HERO), Pharma (SUNPHARMA, CIPLA, DRREDDY, DIVISLAB, APOLLOHOSP), Energy (RELIANCE, ONGC, POWERGRID, NTPC, IOC), FMCG (HINDUNILVR, NESTLEIND, BRITANNIA, DABUR, COLPAL), Metal (TATASTEEL, HINDALCO, JSWSTEEL, VEDL, SAIL), Infra (L&T, ULTRACEMCO, GRASIM, SHREECEM), Defence (HAL, BEL, BHEL, BEML), Consumer (TITAN, TATACONSUM, ASIANPAINT).

### Fetched Metrics
`Name, Sector, Price, Mkt Cap (Cr), P/E, P/B, ROE %, Net Margin %, D/E, Rev Growth %, Div Yield %, Current Ratio, Beta, EPS`

### Filter Operators
| Operator | Meaning |
|----------|---------|
| `<=` | Less than or equal |
| `>=` | Greater than or equal |
| `==` | Exact match |
| `<`  | Strictly less than |
| `>`  | Strictly greater than |

---

## Valuation Models

### 1. DCF (Discounted Cash Flow)
- Uses last 4 years' average Free Cash Flow as base
- Projects 10 years at revenue growth rate (capped at 20%)
- Terminal value using Gordon Growth Model: `TV = FCF₁₀ × (1+g) / (WACC − g)`
- Shares outstanding from `info["sharesOutstanding"]`
- Formula: `Intrinsic = Σ(FCFₙ/(1+WACC)ⁿ) + TV/(1+WACC)¹⁰`

### 2. Graham Number
- `Graham = √(22.5 × EPS × BookValuePerShare)`
- Benjamin Graham's conservative intrinsic value estimate
- Assumes P/E × P/B ≤ 22.5 for a fairly valued stock

### 3. P/E Mean Reversion
- `Fair Value = EPS × Sector_Median_PE`
- Sector PE lookup: Technology=35, Financials=18, Healthcare=28, Industrials=24, Energy=12, default=22
- Assumes price will revert to sector average multiple

### 4. Lynch PEG (in notebook only)
- `PEG = (P/E) / EPS_growth_rate`
- PEG < 1 → Potentially undervalued
- PEG 1–2 → Fair value
- PEG > 2 → Potentially overvalued

---

## Configuration & Usage

### Notebook Configuration
Edit **Cell 2** (`screener.ipynb`):
```python
TICKER  = "RELIANCE.NS"   # ← Your target ticker
PERIOD  = "5y"             # ← Historical data period
BENCH   = "^NSEI"          # ← Benchmark index
```
Then: `Kernel → Restart & Run All`

### App Configuration
All settings via the sidebar. API keys can be entered once and saved to `.env`.

### Ticker Formats
| Market | Format | Example |
|--------|--------|---------|
| NSE India | `SYMBOL.NS` | `HAL.NS` |
| BSE India | `SYMBOL.BO` | `RELIANCE.BO` |
| US Stocks | `SYMBOL` | `AAPL` |
| US ETF | `SYMBOL` | `SPY` |
| Indices | `^SYMBOL` | `^NSEI`, `^GSPC` |

### Performance Notes
- Data is cached for **1 hour** (`@st.cache_data(ttl=3600)`)
- Screener universe (60 stocks) is cached for **30 minutes** (`ttl=1800`)
- Benchmark data cached for **24 hours** (`ttl=86400`)
- First run may take 15–30 seconds depending on internet speed

---

*SCREENER — Python implementation of screener.in × groww.in  ·  Data via Yahoo Finance  ·  Not financial advice*
