"""News scraping from 10+ sources: APIs, RSS, Reddit."""
import warnings
warnings.filterwarnings("ignore")

import time, re
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests
import feedparser
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
}
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def _safe_get(url, params=None, timeout=8):
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _parse_items(raw_list, source: str) -> list[dict]:
    """Normalise raw dict items to {title, url, source, published}."""
    out = []
    for item in raw_list:
        title = item.get("title") or item.get("headline") or item.get("summary", "")[:100]
        url   = item.get("url") or item.get("link") or item.get("newsUrl") or ""
        pub   = item.get("publishedAt") or item.get("published") or item.get("datetime") or ""
        if title:
            out.append({"title": str(title).strip(), "url": url, "source": source, "published": str(pub)})
    return out


# ─── AlphaVantage ─────────────────────────────────────────────────────────────
def fetch_alphavantage(symbol: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    r = _safe_get("https://www.alphavantage.co/query",
                  {"function":"NEWS_SENTIMENT","tickers":symbol,"limit":max_n,"apikey":api_key})
    if not r: return []
    try:
        return _parse_items(r.json().get("feed", [])[:max_n], "AlphaVantage")
    except Exception:
        return []


# ─── Finnhub ──────────────────────────────────────────────────────────────────
def fetch_finnhub(symbol: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    frm = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    to  = datetime.now().strftime("%Y-%m-%d")
    sym = symbol.replace(".NS","").replace(".BO","")
    r   = _safe_get("https://finnhub.io/api/v1/company-news",
                    {"symbol":sym,"from":frm,"to":to,"token":api_key})
    if not r: return []
    try:
        items = r.json()[:max_n]
        return [{"title":i.get("headline",""), "url":i.get("url",""),
                 "source":"Finnhub", "published":str(i.get("datetime",""))} for i in items if i.get("headline")]
    except Exception:
        return []


# ─── Tavily ───────────────────────────────────────────────────────────────────
def fetch_tavily(query: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    try:
        r = _SESSION.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_n, "search_depth": "basic",
                  "api_key": api_key},
            timeout=10
        )
        r.raise_for_status()
        return [{"title": i.get("title",""), "url": i.get("url",""),
                 "source": "Tavily", "published": ""}
                for i in r.json().get("results",[])[:max_n] if i.get("title")]
    except Exception:
        return []


# ─── NewsAPI ─────────────────────────────────────────────────────────────────
def fetch_newsapi(query: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    r = _safe_get("https://newsapi.org/v2/everything",
                  {"q":query,"sortBy":"publishedAt","pageSize":max_n,"apiKey":api_key,"language":"en"})
    if not r: return []
    try:
        return _parse_items(r.json().get("articles",[])[:max_n], "NewsAPI")
    except Exception:
        return []


# ─── EODHD ───────────────────────────────────────────────────────────────────
def fetch_eodhd(symbol: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    sym = symbol.replace(".NS","") + ".NSE" if ".NS" in symbol else symbol
    r   = _safe_get(f"https://eodhd.com/api/news",
                    {"s":sym,"limit":max_n,"api_token":api_key,"fmt":"json"})
    if not r: return []
    try:
        return [{"title": i.get("title",""), "url": i.get("link",""),
                 "source": "EODHD", "published": i.get("date","")}
                for i in r.json()[:max_n] if i.get("title")]
    except Exception:
        return []


# ─── Marketaux ───────────────────────────────────────────────────────────────
def fetch_marketaux(symbol: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    sym = symbol.replace(".NS","").replace(".BO","")
    r   = _safe_get("https://api.marketaux.com/v1/news/all",
                    {"symbols":sym,"filter_entities":"true","language":"en",
                     "limit":max_n,"api_token":api_key})
    if not r: return []
    try:
        return _parse_items(r.json().get("data",[])[:max_n], "Marketaux")
    except Exception:
        return []


# ─── APILayer (World News) ────────────────────────────────────────────────────
def fetch_apilayer(query: str, api_key: str, max_n: int = 15) -> list[dict]:
    if not api_key: return []
    r = _safe_get("https://api.worldnewsapi.com/search-news",
                  {"text":query,"number":max_n,"language":"en","api-key":api_key})
    if not r: return []
    try:
        return _parse_items(r.json().get("news",[])[:max_n], "APILayer")
    except Exception:
        return []


# ─── Yahoo RSS ────────────────────────────────────────────────────────────────
def fetch_yahoo_rss(symbol: str, max_n: int = 15) -> list[dict]:
    url  = f"https://finance.yahoo.com/rss/headline?s={quote_plus(symbol)}"
    feed = feedparser.parse(url)
    return [{"title": e.title, "url": e.link, "source": "Yahoo RSS",
             "published": e.get("published","")}
            for e in feed.entries[:max_n] if e.get("title")]


# ─── GNews ───────────────────────────────────────────────────────────────────
def fetch_gnews_rss(query: str, max_n: int = 15) -> list[dict]:
    url  = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [{"title": e.title, "url": e.link, "source": "GNews",
             "published": e.get("published","")}
            for e in feed.entries[:max_n] if e.get("title")]


# ─── ET Markets RSS ───────────────────────────────────────────────────────────
def fetch_et_markets_rss(max_n: int = 15) -> list[dict]:
    url  = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    feed = feedparser.parse(url)
    return [{"title": e.title, "url": e.link, "source": "ET Markets",
             "published": e.get("published","")}
            for e in feed.entries[:max_n] if e.get("title")]


# ─── Livemint RSS ─────────────────────────────────────────────────────────────
def fetch_livemint_rss(max_n: int = 15) -> list[dict]:
    url  = "https://www.livemint.com/rss/markets"
    feed = feedparser.parse(url)
    return [{"title": e.title, "url": e.link, "source": "Livemint",
             "published": e.get("published","")}
            for e in feed.entries[:max_n] if e.get("title")]


# ─── Reddit RSS ───────────────────────────────────────────────────────────────
def fetch_reddit_rss(search_query: str, max_n: int = 15) -> list[dict]:
    subs  = ["IndiaInvestments","StockMarket","wallstreetbets","investing","stocks","IndianStockMarket"]
    items = []
    q     = quote_plus(search_query)
    for sub in subs:
        url  = f"https://www.reddit.com/r/{sub}/search.rss?q={q}&restrict_sr=1&sort=new"
        feed = feedparser.parse(url)
        for e in feed.entries[:3]:
            if e.get("title"):
                items.append({"title": e.title, "url": e.link,
                              "source": f"Reddit/{sub}", "published": e.get("published","")})
        if len(items) >= max_n: break
    return items[:max_n]


# ─── StockTwits (US only) ─────────────────────────────────────────────────────
def fetch_stocktwits(symbol: str, max_n: int = 15) -> list[dict]:
    if symbol.endswith((".NS",".BO")):
        return []   # Indian stocks not on StockTwits
    sym = symbol.replace("^","")
    r   = _safe_get(f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json")
    if not r: return []
    try:
        msgs = r.json().get("messages",[])[:max_n]
        return [{"title": m.get("body","")[:200], "url": "",
                 "source": "StockTwits", "published": m.get("created_at","")}
                for m in msgs if m.get("body")]
    except Exception:
        return []


# ─── Master fetch + dedup ─────────────────────────────────────────────────────
def fetch_all_news(symbol: str, company_name: str, api_keys: dict, max_per_source: int = 15) -> list[dict]:
    """Run all scrapers and return deduplicated list."""
    us_sym    = symbol.replace(".NS","").replace(".BO","")
    search_q  = company_name or us_sym
    all_items = []

    # API-based
    all_items += fetch_alphavantage(us_sym, api_keys.get("ALPHA_VANTAGE",""), max_per_source)
    all_items += fetch_finnhub(us_sym, api_keys.get("FINNHUB",""), max_per_source)
    all_items += fetch_tavily(search_q, api_keys.get("TAVILY",""), max_per_source)
    all_items += fetch_newsapi(search_q, api_keys.get("NEWSAPI",""), max_per_source)
    all_items += fetch_eodhd(symbol, api_keys.get("EODHD",""), max_per_source)
    all_items += fetch_marketaux(symbol, api_keys.get("MARKETAUX",""), max_per_source)
    all_items += fetch_apilayer(search_q, api_keys.get("APILAYER",""), max_per_source)

    # RSS-based
    all_items += fetch_yahoo_rss(symbol, max_per_source)
    all_items += fetch_gnews_rss(search_q, max_per_source)
    all_items += fetch_et_markets_rss(max_per_source)
    all_items += fetch_livemint_rss(max_per_source)
    all_items += fetch_reddit_rss(search_q, max_per_source)
    all_items += fetch_stocktwits(symbol, max_per_source)

    # Dedup by title
    seen, unique = set(), []
    kw = (company_name or "").lower().split()[0] if company_name else us_sym.lower()
    for item in all_items:
        t = item.get("title","").strip().lower()
        if not t or t in seen: continue
        seen.add(t)
        unique.append(item)

    # Filter relevant
    company_kw = [w.lower() for w in (company_name or "").split() if len(w) > 3]
    company_kw += [us_sym.lower(), symbol.lower()]
    relevant = [i for i in unique if
                any(kw in i.get("title","").lower() for kw in company_kw)]

    return relevant if relevant else unique[:50]
