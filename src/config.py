"""
config.py
---------
Central configuration for Market Sentiment Analyzer.
All constants, defaults, and paths live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CACHE_DIR  = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR    = ROOT_DIR / "logs"

for _d in (CACHE_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
NEWSAPI_BASE  = "https://newsapi.org/v2/everything"
GNEWS_BASE = "https://gnews.io/api/v4/search"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
NEWSAPI_PAGE_SIZE   = 20
NEWSAPI_FAST_CHUNK_DAYS = 7
NEWSAPI_MAX_ARTICLES = 120          # keep free-tier usage practical
NEWSAPI_TIMEOUT      = 12           # seconds
NEWSAPI_RETRY_TIMES  = 1
NEWSAPI_RETRY_WAIT   = 2            # seconds between retries

NEWS_PROVIDERS = [
    "newsapi",
    "gnews",
    "google_news",
    "yahoo_rss",
    "cnbc_rss",
    "marketwatch_rss",
]

# ── Yahoo Finance fallback ────────────────────────────────────────
YAHOO_RSS_URLS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    "https://finance.yahoo.com/rss/topstories",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
]
RSS_PROVIDER_URLS = {
    "yahoo_rss": YAHOO_RSS_URLS,
    "cnbc_rss": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ],
    "marketwatch_rss": [
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    ],
}
YAHOO_TIMEOUT = 10

# ── Market data ───────────────────────────────────────────────────
DEFAULT_TICKER   = "^GSPC"
MARKET_INDEX_OPTIONS = {
    "^GSPC": "S&P 500 Index",
    "^NDX": "Nasdaq 100 Index",
    "^IXIC": "Nasdaq Composite Index",
}
MARKET_TIMEOUT   = 15
MARKET_DATE_BUFFER = 14             # extra calendar days to ensure enough trading days
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "fred").lower()
MARKET_PROVIDER_PRIORITY = ["fred", "alpha_vantage", "stooq", "yfinance"]
FRED_DOWNLOAD_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SYMBOL_MAP = {
    "^GSPC": "SP500",
    "^SPX": "SP500",
    "^NDX": "NASDAQ100",
    "^IXIC": "NASDAQCOM",
}
ALPHA_VANTAGE_SYMBOL_MAP = {
    "^GSPC": "SPY",
    "^SPX": "SPY",
    "^NDX": "QQQ",
    "^IXIC": "QQQ",
}
STOOQ_DOWNLOAD_URL = "https://stooq.com/q/d/l/"

STOOQ_SYMBOL_MAP = {
    "^GSPC": "^spx",
    "^SPX": "^spx",
    "^NDX": "^ndx",
    "^DJI": "^dji",
    "^IXIC": "^ixic",
    "SPY": "spy.us",
    "QQQ": "qqq.us",
    "DIA": "dia.us",
    "IWM": "iwm.us",
}

# ── Sentiment ─────────────────────────────────────────────────────
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05
SENTIMENT_MA_WINDOW = 5

# ── Correlation ───────────────────────────────────────────────────
MAX_LAG_DAYS     = 5
ROLLING_WINDOW   = 7
SIGNIFICANCE_LVL = 0.05
MIN_OBSERVATIONS = 5                # min data points for any stat test

# ── NLP / Text cleaning ───────────────────────────────────────────
URL_PATTERN     = r"https?://\S+"
HTML_TAG_PATTERN = r"<[^>]+>"
PUNCT_EXTRA     = r"[^\w\s\-\']"    # characters to strip (keep hyphens, apostrophes)
MIN_TEXT_LEN    = 5                 # discard text shorter than this after cleaning

# ── Caching ───────────────────────────────────────────────────────
NEWS_CACHE_TTL_HOURS    = 24        # re-fetch if older than this
MARKET_CACHE_TTL_HOURS  = 1

# ── Sector → NewsAPI query mapping ────────────────────────────────
SECTOR_QUERIES = {
    "tech":     "technology OR stocks OR earnings OR Apple OR Microsoft OR Nvidia",
    "finance":  "banks OR Federal Reserve OR interest rates OR JPMorgan OR Goldman",
    "energy":   "oil OR energy OR crude OR OPEC OR Exxon OR Chevron",
    "health":   "healthcare OR pharmaceutical OR biotech OR FDA OR Pfizer OR Moderna",
    "all":      "S&P 500 OR stock market OR Wall Street OR earnings OR economy",
}

# ── Palette (shared across visualizer + dashboard) ───────────────
PALETTE = {
    "sentiment": "#1baf7a",
    "spx":       "#2a78d6",
    "positive":  "#1baf7a",
    "negative":  "#e34948",
    "neutral":   "#eda100",
    "ma":        "#4a3aa7",
    "bg":        "#0f1117",
    "card":      "#1e2130",
    "text":      "#e8eaf0",
}

# ── Logging ───────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = LOG_DIR / "analyzer.log"
