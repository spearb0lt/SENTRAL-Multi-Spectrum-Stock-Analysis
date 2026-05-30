"""ML forecasting: LSTM, Transformer, ensemble, Monte Carlo GBM."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─── Models ───────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _make_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(targets[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_model(model, X_train, y_train, epochs, batch_size, lr=1e-3,
                  progress_cb=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.HuberLoss()
    model.train()
    losses = []
    for ep in range(1, epochs + 1):
        perm  = torch.randperm(len(X_train))
        ep_loss = 0
        for i in range(0, len(X_train), batch_size):
            idx   = perm[i:i+batch_size]
            xb    = X_train[idx]; yb = y_train[idx]
            optimizer.zero_grad()
            pred  = model(xb)
            loss  = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        avg = ep_loss / max(1, len(X_train) // batch_size)
        losses.append(avg)
        if progress_cb and ep % 10 == 0:
            progress_cb(ep, epochs, avg)
    return losses


def _predict(model, X_test_t, scaler_close):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    return scaler_close.inverse_transform(preds.reshape(-1, 1)).flatten()


def _forecast_future(model, last_seq, scaler_close, scaler_feat, n_days):
    model.eval()
    seq = last_seq.copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_days):
            inp  = torch.tensor(seq[-1:], dtype=torch.float32)
            out  = model(inp).item()
            preds.append(out)
            new_row        = seq[-1, -1].copy() if seq.ndim == 3 else seq[-1].copy()
            new_row        = seq[0, -1].copy()
            new_row[0]     = out   # Close (scaled) as first feature
            seq = np.concatenate([seq[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
    return scaler_close.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# ─── Main training entry point ────────────────────────────────────────────────
def run_ml_forecast(df: pd.DataFrame, forecast_days: int = 30,
                     seq_len: int = 60, epochs: int = 50,
                     batch_size: int = 32, hidden_size: int = 64,
                     n_layers: int = 2,
                     progress_cb=None) -> dict:
    """
    Train LSTM + Transformer and forecast forecast_days ahead.
    Returns dict with forecasts, metrics, scalers.
    """
    feature_cols = ["Close", "RSI", "MACD_Hist", "ATR", "BB_Pct"]
    feat_df = df[feature_cols].dropna()

    scaler_close = MinMaxScaler()
    scaler_feat  = MinMaxScaler()
    feat_scaled  = scaler_feat.fit_transform(feat_df.values)
    close_scaled = scaler_close.fit_transform(feat_df[["Close"]].values)

    X, y    = _make_sequences(feat_scaled, close_scaled, seq_len)
    split   = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_test  = torch.tensor(X[split:])
    y_test  = y[split:]
    test_dates = feat_df.index[seq_len + split:]

    n_features = X.shape[2]

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if progress_cb: progress_cb("lstm_start", 0, 0)
    lstm_model = LSTMModel(n_features, hidden_size, n_layers)
    _train_model(lstm_model, X_train, y_train, epochs, batch_size,
                  progress_cb=lambda ep,tot,loss: (progress_cb("lstm_epoch", ep, tot) if progress_cb else None))
    lstm_preds  = _predict(lstm_model, X_test, scaler_close)
    actuals     = scaler_close.inverse_transform(y_test.reshape(-1,1)).flatten()
    lstm_mae    = float(mean_absolute_error(actuals, lstm_preds))
    lstm_rmse   = float(np.sqrt(mean_squared_error(actuals, lstm_preds)))
    lstm_mape   = float(np.mean(np.abs((actuals - lstm_preds) / (actuals + 1e-8))) * 100)

    # ── Transformer ───────────────────────────────────────────────────────────
    if progress_cb: progress_cb("trans_start", 0, 0)
    trans_model = TemporalTransformer(n_features)
    _train_model(trans_model, X_train, y_train, epochs, batch_size,
                  progress_cb=lambda ep,tot,loss: (progress_cb("trans_epoch", ep, tot) if progress_cb else None))
    trans_preds = _predict(trans_model, X_test, scaler_close)
    trans_mae   = float(mean_absolute_error(actuals, trans_preds))
    trans_rmse  = float(np.sqrt(mean_squared_error(actuals, trans_preds)))
    trans_mape  = float(np.mean(np.abs((actuals - trans_preds) / (actuals + 1e-8))) * 100)

    # ── Future forecast ───────────────────────────────────────────────────────
    last_seq     = feat_scaled[-seq_len:].reshape(1, seq_len, n_features)
    lstm_future  = _forecast_future(lstm_model, last_seq, scaler_close, scaler_feat, forecast_days)
    trans_future = _forecast_future(trans_model, last_seq, scaler_close, scaler_feat, forecast_days)
    ensemble_future = (lstm_future + trans_future) / 2

    current_price = float(df["Close"].iloc[-1])
    last_date     = df.index[-1]
    future_dates  = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    eval_df = pd.DataFrame({
        "MAE":    [round(lstm_mae,2), round(trans_mae,2)],
        "RMSE":   [round(lstm_rmse,2), round(trans_rmse,2)],
        "MAPE %": [f"{lstm_mape:.2f}%", f"{trans_mape:.2f}%"],
    }, index=pd.Index(["LSTM","Transformer"], name="Model"))

    return {
        "lstm_model": lstm_model, "trans_model": trans_model,
        "scaler_close": scaler_close, "scaler_feat": scaler_feat,
        "lstm_preds": lstm_preds, "trans_preds": trans_preds,
        "actuals": actuals, "test_dates": test_dates,
        "lstm_future": lstm_future, "trans_future": trans_future,
        "ensemble_future": ensemble_future, "future_dates": future_dates,
        "current_price": current_price, "eval_df": eval_df,
        "lstm_30d": float(lstm_future[-1]),
        "trans_30d": float(trans_future[-1]),
        "ensemble_30d": float(ensemble_future[-1]),
        "forecast_return": (float(ensemble_future[-1]) / current_price) - 1,
    }


# ─── Monte Carlo ──────────────────────────────────────────────────────────────
def run_monte_carlo(close: pd.Series, forecast_days: int = 30,
                     n_sims: int = 1000, seed: int = 42) -> dict:
    np.random.seed(seed)
    log_rets  = np.log(close / close.shift(1)).dropna()
    mu        = log_rets.mean()
    sig       = log_rets.std()
    S0        = float(close.iloc[-1])

    paths = np.zeros((n_sims, forecast_days + 1))
    paths[:, 0] = S0
    for t in range(1, forecast_days + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sig**2) + sig*Z)

    terminal  = paths[:, -1]
    pctiles   = {p: float(np.percentile(terminal, p)) for p in [5,10,25,50,75,90,95]}
    pct_paths = {p: np.percentile(paths, p, axis=0) for p in [5,25,50,75,95]}

    return {
        "paths": paths, "terminal": terminal,
        "pctiles": pctiles, "pct_paths": pct_paths,
        "S0": S0, "mu_ann": mu*252, "sig_ann": sig*252**0.5,
        "p_profit": float((terminal > S0).mean()),
        "p_gain5":  float((terminal > S0*1.05).mean()),
        "p_loss5":  float((terminal < S0*0.95).mean()),
        "n_sims": n_sims, "days": forecast_days,
    }
