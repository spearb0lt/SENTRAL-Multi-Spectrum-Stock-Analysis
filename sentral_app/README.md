# SENTRAL Streamlit App

A full Streamlit web interface that mirrors everything in `SENTRAL_Complete.ipynb`.

## Structure

```
sentral_app/
├── app.py                     # Main Streamlit app — entry point
├── requirements_app.txt       # Additional pip packages for the app
└── modules/
    ├── data_loader.py         # yfinance download + technical features
    ├── fundamental.py         # DCF, Altman Z, Piotroski F, metrics
    ├── technical.py           # Risk metrics, patterns, seasonality
    ├── peers.py               # Sector peer comparison
    ├── news.py                # 12+ news sources (APIs + RSS + Reddit)
    ├── sentiment.py           # 10 AI sentiment models + ensemble
    ├── ml_forecast.py         # LSTM, Transformer, Monte Carlo
    ├── signals.py             # Signal engine (BUY / HOLD / SELL)
    ├── backtest.py            # 20 strategies + Kelly criterion
    └── report.py              # HTML + PDF report generation
```

## Setup

```bash
# 1. Activate venv (same as notebook)
..\venv\Scripts\activate

# 2. Install extra deps
pip install streamlit reportlab python-dotenv

# 3. Run the app
cd sentral_app
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Usage

1. Enter ticker, period, and forecast settings in the **sidebar**
2. Paste API keys (or load from existing `.env`)
3. Optionally check **Save keys to .env** to persist them
4. Click **Run Full Analysis**
5. Wait ~2-5 min for full run (ML training is the slowest part)
6. Browse results in tabs; download HTML or PDF report

## Tabs

| Tab | Content |
|---|---|
| 📊 Signal | BUY/HOLD/SELL gauge + pillar breakdown + risk flags |
| 💰 Fundamentals | Metrics, DCF, Altman Z, Piotroski F |
| 📉 Technical | Candlestick chart, indicators, patterns, seasonality |
| 🗞️ News & Sentiment | 10-model sentiment + news table |
| 🧠 ML Forecast | LSTM + Transformer forecasts + Monte Carlo |
| 📈 Backtest | 20 strategies + Kelly position sizing |
| 👥 Peers | Sector peer comparison |
| 📰 Thesis | Gemini-generated investment thesis |
