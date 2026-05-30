"""Sentiment analysis: 10 models (VADER, HuggingFace, Groq, Gemini)."""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np

# ─── helpers ─────────────────────────────────────────────────────────────────
def _map_label(raw: str) -> str:
    r = raw.lower()
    if any(x in r for x in ["positive","pos","bullish","buy","up","label_2","label_1"]):
        return "positive"
    if any(x in r for x in ["negative","neg","bearish","sell","down","label_0"]):
        return "negative"
    return "neutral"

def _score(label: str) -> float:
    return {"positive":1.0, "neutral":0.0, "negative":-1.0}.get(label, 0.0)


# ─── VADER ────────────────────────────────────────────────────────────────────
def run_vader(texts: list[str]) -> list[tuple]:
    try:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        out = []
        for t in texts:
            c = sia.polarity_scores(t)["compound"]
            lbl = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
            out.append((lbl, abs(c)))
        return out
    except Exception:
        return []


# ─── HuggingFace Inference API ────────────────────────────────────────────────
_HF_BASE = "https://api-inference.huggingface.co/models"

def _hf_classify(texts: list[str], model_id: str, hf_token: str,
                  max_items: int = 10) -> list[tuple]:
    """Call HF inference API and return list of (label, confidence)."""
    if not hf_token: return []
    try:
        import requests
        headers = {"Authorization": f"Bearer {hf_token}"}
        results = []
        for t in texts[:max_items]:
            url = f"{_HF_BASE}/{model_id}"
            r = requests.post(url, headers=headers, json={"inputs": t[:512]}, timeout=15)
            if r.status_code != 200: continue
            preds = r.json()
            if isinstance(preds, list) and isinstance(preds[0], list):
                preds = preds[0]
            if not preds: continue
            best = max(preds, key=lambda x: x.get("score", 0))
            lbl  = _map_label(best.get("label","neutral"))
            results.append((lbl, float(best.get("score", 0.5))))
            time.sleep(0.15)
        return results
    except Exception:
        return []


def run_finbert(texts, hf_token):
    return _hf_classify(texts, "ProsusAI/finbert", hf_token)


def run_finbert_tone(texts, hf_token):
    res = _hf_classify(texts, "yiyanghkust/finbert-tone:fastest", hf_token)
    if not res or all(_score(r[0]) == 0.0 for r in res):
        res = _hf_classify(texts, "cardiffnlp/twitter-roberta-base-sentiment-latest", hf_token)
    return res


def run_distilroberta(texts, hf_token):
    return _hf_classify(texts, "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", hf_token)


def run_roberta_large(texts, hf_token):
    return _hf_classify(texts, "siebert/sentiment-roberta-large-english", hf_token)


def run_stocktwits_roberta(texts, hf_token):
    res = _hf_classify(texts, "zhayunduo/roberta-base-stocktwits-finetuned", hf_token)
    if not res or all(_score(r[0]) == 0.0 for r in res):
        return [("neutral", 0.5)] * len(texts[:10])
    return res


# ─── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_MODELS_PRIORITY = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = ("You are a financial sentiment classifier. "
                 "Reply with exactly ONE word: positive, negative, or neutral.")

def _groq_classify_one(text: str, groq_key: str, model: str) -> tuple | None:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp   = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":f"Headline: {text[:400]}"}],
            max_tokens=3, temperature=0
        )
        lbl = resp.choices[0].message.content.strip().lower()
        lbl = _map_label(lbl)
        return (lbl, 1.0)
    except Exception:
        return None

def run_groq(texts: list[str], groq_key: str, model_slot: int = 0,
             max_items: int = 8) -> tuple[list[tuple], str]:
    """Run Groq sentiment. model_slot 0=primary, 1=secondary."""
    if not groq_key: return [], ""
    model_list = GROQ_MODELS_PRIORITY[model_slot:] + GROQ_MODELS_PRIORITY[:model_slot]
    for model in model_list:
        results = []
        for t in texts[:max_items]:
            r = _groq_classify_one(t, groq_key, model)
            results.append(r if r else ("neutral", 0.5))
            time.sleep(0.05)
        if results and not all(_score(r[0]) == 0.0 for r in results):
            return results, model
    return [], ""


# ─── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_MODELS_LIST = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]

def _detect_genai_sdk():
    try:
        import google.genai as g
        return g, "new"
    except ImportError:
        pass
    try:
        import google.generativeai as g
        return g, "old"
    except ImportError:
        return None, None

def _gemini_classify_batch(texts: list[str], model_id: str,
                            gemini_key: str, label: str) -> list[tuple] | None:
    if not gemini_key: return None
    sdk, ver = _detect_genai_sdk()
    if sdk is None: return None
    prompt = (
        "Classify the sentiment of each headline as positive, negative, or neutral. "
        "Respond with ONLY a comma-separated list of labels in the same order.\n\n"
        + "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(texts))
    )
    try:
        if ver == "new":
            client = sdk.Client(api_key=gemini_key)
            resp   = client.models.generate_content(model=model_id, contents=prompt)
            raw    = resp.text.strip()
        else:
            sdk.configure(api_key=gemini_key)
            m   = sdk.GenerativeModel(model_id)
            raw = m.generate_content(prompt).text.strip()

        labels = [_map_label(x.strip()) for x in raw.split(",")]
        while len(labels) < len(texts):
            labels.append("neutral")
        return [(labels[i], 1.0) for i in range(len(texts))]
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err: return None
        return None


def run_gemini(texts: list[str], gemini_key: str,
               skip_model: str = "", max_items: int = 8) -> tuple[list[tuple], str]:
    if not gemini_key: return [], ""
    for m in GEMINI_MODELS_LIST:
        if m == skip_model: continue
        res = _gemini_classify_batch(texts[:max_items], m, gemini_key, f"Gemini/{m}")
        if res: return res, m
    return [], ""


# ─── Generate Investment Thesis ───────────────────────────────────────────────
def generate_thesis(headlines: list[str], sentiment_df,
                    ticker: str, company: str,
                    ensemble_score: float, gemini_key: str) -> str:
    """Generate AI investment thesis using best available Gemini model."""
    if not gemini_key:
        return "⚠ No GEMINI_KEY provided — thesis skipped."

    top   = "\n".join(f"- {h}" for h in headlines[:15])
    scores = ""
    if hasattr(sentiment_df, "to_dict"):
        try:
            scores = "\n".join(f"  {m}: {s:+.3f}" for m,s in sentiment_df["avg_score"].to_dict().items())
        except Exception:
            pass

    prompt = (
        f"You are a senior equity research analyst. Write a concise 200-word investment thesis "
        f"for {company} ({ticker}).\n\n"
        f"RECENT NEWS HEADLINES:\n{top}\n\n"
        f"AI SENTIMENT SCORES (scale -1 to +1):\n{scores}\n\n"
        f"ENSEMBLE SENTIMENT: {ensemble_score:+.3f}\n\n"
        "- Identify dominant narrative\n- Highlight key risks and catalysts\n"
        "- State BUY/HOLD/SELL stance\n- Be concise and professional.\n"
        "Add a brief disclaimer at the end only."
    )
    sdk, ver = _detect_genai_sdk()
    if sdk is None:
        return "⚠ google-generativeai package not installed."

    for m in GEMINI_MODELS_LIST:
        try:
            if ver == "new":
                client = sdk.Client(api_key=gemini_key)
                text   = client.models.generate_content(model=m, contents=prompt).text.strip()
            else:
                sdk.configure(api_key=gemini_key)
                text = sdk.GenerativeModel(m).generate_content(prompt).text.strip()
            if text:
                return f"*(Generated by {m})*\n\n{text}"
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            return f"Error generating thesis: {e}"
    return "⚠ All Gemini models quota-exhausted (free tier: ~50 req/day)."


# ─── Ensemble aggregation ─────────────────────────────────────────────────────
def aggregate_sentiment(results: dict) -> tuple:
    """
    Returns (df_summary, ensemble_score, overall_label)
    results: {model_name: [(label, confidence), ...]}
    """
    import pandas as pd
    summaries = {}
    for model, res in results.items():
        if not res: continue
        labels = [r[0] for r in res]
        scores = [_score(l) for l in labels]
        summaries[model] = {
            "avg_score":  round(float(np.mean(scores)), 3),
            "positive%":  round(labels.count("positive") / len(labels) * 100, 1),
            "neutral%":   round(labels.count("neutral")  / len(labels) * 100, 1),
            "negative%":  round(labels.count("negative") / len(labels) * 100, 1),
            "n_articles": len(res),
        }
    df = pd.DataFrame(summaries).T if summaries else pd.DataFrame()
    ens = float(df["avg_score"].mean()) if not df.empty else 0.0
    label = "POSITIVE" if ens > 0.1 else ("NEGATIVE" if ens < -0.1 else "NEUTRAL")
    return df, ens, label
