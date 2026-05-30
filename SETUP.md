# SETUP — Getting Started with SENTRAL × SCREENER

> **Step-by-step guide to set up the environment, configure API keys, and run both Streamlit apps and notebooks.**

---

## Prerequisites

| Requirement | Version | Check |
|------------|---------|-------|
| Python | ≥ 3.10 | `python --version` |
| pip | ≥ 23.0 | `pip --version` |
| Git | Any | `git --version` |
| ~3 GB disk space | For ML models + data | — |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/SENTRAL-Multi-Spectrum-Stock-Analysis
cd SENTRAL-Multi-Spectrum-Stock-Analysis
```

---

## Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏳ This installs PyTorch, Transformers, and other large packages — may take **5–15 minutes** on first install.

### GPU Support (Optional)
For CUDA-accelerated training (faster LSTM/Transformer):
1. Check your CUDA version: `nvidia-smi`
2. Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
3. Get the correct pip command for your CUDA version, e.g.:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Step 4 — Configure API Keys

Create a `.env` file in the repository root:

```bash
# Windows
copy NUL .env

# Linux / macOS
touch .env
```

Open `.env` and add your keys:

```env
# ── Required for LLM peer discovery ────────────────────────
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=AIzaSy_your_key_here

# ── Required for ML sentiment models ───────────────────────
HF_TOKEN=hf_your_token_here

# ── Optional: News APIs (more sources = better sentiment) ──
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
TAVILY_API_KEY=
NEWSAPI_KEY=
EODHD_API_KEY=
MARKETAUX_API_KEY=
APILAYER_API_KEY=

# ── Optional: Reddit OAuth (better sentiment quality) ──────
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USERNAME=
```

### Getting Free API Keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Groq** (best for peers) | [console.groq.com](https://console.groq.com) | Very generous |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) | Free tier |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | 25 req/day |
| Finnhub | [finnhub.io](https://finnhub.io/register) | 60 req/min |
| NewsAPI | [newsapi.org/register](https://newsapi.org/register) | 100 req/day |
| Tavily | [tavily.com](https://tavily.com) | 1000 req/month |
| Reddit OAuth | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free |

> **Minimum to run:** Groq key for LLM peer discovery. The apps work without any keys (falls back to hard-coded peer lists and skips LLM sentiment), but Groq + Gemini + HuggingFace dramatically improve results.

---

## Step 5 — Register the Jupyter Kernel

To use the notebooks in VS Code or JupyterLab:

```bash
# Activate venv first (see Step 2)
python -m ipykernel install --user --name sentral-venv --display-name "Python (SENTRAL venv)"
```

In VS Code: open any `.ipynb` → click the kernel selector (top right) → choose **Python (SENTRAL venv)**.

---

## Step 6 — Run the Notebooks

### SENTRAL_Complete.ipynb (Full 8-pillar analysis)
1. Open `SENTRAL_Complete.ipynb` in VS Code
2. Select kernel: **Python (SENTRAL venv)**
3. Edit Cell 2 to set your ticker:
   ```python
   TICKER = "HAL.NS"   # Change me
   PERIOD = "5y"
   ```
4. `Kernel → Restart & Run All` (takes 3–10 minutes with ML enabled)

### screener.ipynb (screener.in × groww.in replica)
1. Open `screener.ipynb`
2. Select kernel: **Python (SENTRAL venv)**
3. Edit Cell 2:
   ```python
   TICKER = "RELIANCE.NS"
   PERIOD = "5y"
   BENCH  = "^NSEI"
   ```
4. `Kernel → Restart & Run All`

---

## Step 7 — Run the Streamlit Apps

### SENTRAL App

```bash
# Windows
cd sentral_app
..\venv\Scripts\streamlit run app.py

# Linux / macOS
cd sentral_app
../venv/bin/streamlit run app.py
```

Opens at: **http://localhost:8501**

### SCREENER App

Open a **second terminal** (keep SENTRAL running):

```bash
# Windows
cd screener_app
..\venv\Scripts\streamlit run app.py --server.port 8502

# Linux / macOS
cd screener_app
../venv/bin/streamlit run app.py --server.port 8502
```

Opens at: **http://localhost:8502**

---

## Typical Ticker Formats

| Market | Format | Examples |
|--------|--------|---------|
| NSE India | `TICKER.NS` | `HAL.NS`, `RELIANCE.NS`, `TCS.NS` |
| BSE India | `TICKER.BO` | `RELIANCE.BO`, `TATAMOTORS.BO` |
| US Stocks | `TICKER` | `AAPL`, `MSFT`, `TSLA` |
| US ETFs | `TICKER` | `SPY`, `QQQ`, `VTI` |
| NSE Index | `^NSEI` | Nifty 50 |
| S&P 500 | `^GSPC` | S&P 500 |
| BSE Sensex | `^BSESN` | BSE Sensex |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ta'`
```bash
pip install ta
```

### `AttributeError: module 'ta.volume' has no attribute 'VolumeWeightedAveragePriceIndicator'`
This is fixed in the latest code. The correct class name is `VolumeWeightedAveragePrice` in ta 0.11.x.

### `UnserializableReturnValueError` in Streamlit
Fixed in the latest code — `yfinance.Ticker` objects are not included in `@st.cache_data` return values.

### `torch` not found or import fails
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Slow first run (ML models)
HuggingFace models (FinBERT, RoBERTa) are downloaded on first use and cached in `~/.cache/huggingface/`. Subsequent runs are instant.

### `st.cache_data` warning about unhashable argument
The `screener_app` passes ticker lists as `tuple` to `fetch_screener_data()` — this is intentional. If you see warnings, ensure the list is converted to `tuple` before passing.

### No news articles found
- Check API keys are correctly set in `.env`
- RSS feeds (Yahoo, Google News) work without keys — if all fail, check internet connectivity
- Finnhub free tier requires a small delay between requests

### yfinance data errors for Indian stocks
- NSE tickers require `.NS` suffix: `HAL` → `HAL.NS`
- Some BSE tickers work better with `.BO` suffix
- Use `yfinance.Ticker("HAL.NS").info` in a Python shell to test

### LSTM training is slow
- Enable GPU: see GPU Support section above
- Reduce `N_SIMULATIONS` to 500 and `LOOKBACK` to 30 in notebook Cell 2
- Set `USE_ML = False` to skip ML forecasting entirely

---

## Directory Structure After Setup

```
SENTRAL-Multi-Spectrum-Stock-Analysis/
├── venv/                       ← Virtual environment (not committed)
├── .env                        ← API keys (not committed)
├── outputs/                    ← Generated outputs per run
│   └── HAL.NS_20260531_0016/
│       ├── signal_report.json
│       ├── forecast_30d.csv
│       └── ...
├── SENTRAL_Complete.ipynb
├── screener.ipynb
├── sentral_app/
│   └── app.py
├── screener_app/
│   └── app.py
├── requirements.txt
├── README.md
├── SENTRAL.md
├── SCREENER.md
└── SETUP.md                    ← You are here
```

---

## Updating

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

*SENTRAL × SCREENER  ·  Setup Guide  ·  Not financial advice*
