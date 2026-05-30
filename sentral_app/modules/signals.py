"""Signal engine: pillar scores and final BUY/HOLD/SELL signal."""
import numpy as np

WEIGHTS = {"Fundamental": 0.30, "Technical": 0.30, "Sentiment": 0.20, "Forecast": 0.20}


def compute_signal(fund_score: float, tech_score: float,
                   sent_score: float, forecast_return: float) -> dict:
    """Compute composite signal from four pillar scores."""
    forecast_score = float(np.clip(forecast_return * 5, -1, 1))

    pillar_scores = {
        "Fundamental": round(fund_score, 4),
        "Technical":   round(tech_score, 4),
        "Sentiment":   round(sent_score, 4),
        "Forecast":    round(forecast_score, 4),
    }

    final_score = sum(pillar_scores[k] * WEIGHTS[k] for k in WEIGHTS)

    if   final_score >=  0.20: signal = "BUY 🟢";  signal_color = "green"
    elif final_score <= -0.20: signal = "SELL 🔴"; signal_color = "red"
    else:                      signal = "HOLD 🟡"; signal_color = "goldenrod"

    confidence_pct = round(min((abs(final_score) / 1.0) * 100, 100), 1)

    return {
        "pillar_scores":   pillar_scores,
        "final_score":     round(final_score, 4),
        "signal":          signal,
        "signal_color":    signal_color,
        "confidence_pct":  confidence_pct,
    }
