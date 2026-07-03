"""
news_fetcher.py
---------------
Multi-provider financial news fetcher.

Providers are queried concurrently, then merged and deduplicated. NewsAPI and
GNews are optional keyed providers; RSS providers keep the app useful when API
quota is exhausted.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cache import cache_get, cache_set
from config import (
    GNEWS_API_KEY,
    GNEWS_BASE,
    HTML_TAG_PATTERN,
    MIN_TEXT_LEN,
    NEWS_API_KEY,
    NEWS_PROVIDERS,
    NEWSAPI_BASE,
    NEWSAPI_FAST_CHUNK_DAYS,
    NEWSAPI_MAX_ARTICLES,
    NEWSAPI_PAGE_SIZE,
    NEWSAPI_RETRY_TIMES,
    NEWSAPI_TIMEOUT,
    NEWS_CACHE_TTL_HOURS,
    RSS_PROVIDER_URLS,
    URL_PATTERN,
    YAHOO_TIMEOUT,
)
from logger import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class NewsFetcher:
    """Fetch and clean financial headlines from multiple providers."""

    def __init__(self, api_key: str = "", use_cache: bool = True):
        self.api_key = api_key or NEWS_API_KEY
        self.gnews_key = GNEWS_API_KEY
        self.use_cache = use_cache
        self.last_source = ""
        self.provider_stats: dict[str, dict] = {}
        self.duplicates_removed = 0
        self.cache_hit = False

    def fetch(
        self,
        query: str,
        days: int = 30,
        end_date: datetime | pd.Timestamp | None = None,
        fast_mode: bool = True,
    ) -> pd.DataFrame:
        """Fetch headlines over the requested date window."""
        end_label = pd.Timestamp(end_date).strftime("%Y-%m-%d") if end_date is not None else "latest"
        mode_label = "fast" if fast_mode else "full"
        cache_key = f"news::multi::v1::{mode_label}::{query}::{days}::{end_label}"

        self.cache_hit = False
        self.provider_stats = {}
        self.duplicates_removed = 0
        self.last_source = ""

        if self.use_cache:
            cached = cache_get(cache_key)
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                self.cache_hit = True
                self.last_source = "cache"
                self._record_provider("cache", len(cached), "hit", "")
                log.info("News loaded from cache (%d articles)", len(cached))
                return cached

        df = self._fetch_from_source(query, days, end_date=end_date, fast_mode=fast_mode)
        df = self._clean(df)

        if df.empty:
            log.warning("No articles returned for query '%s'", query)
        else:
            log.info("Fetched %d articles for query '%s'", len(df), query)
            if self.use_cache:
                cache_set(cache_key, df, ttl_hours=NEWS_CACHE_TTL_HOURS)

        return df

    def _fetch_from_source(
        self,
        query: str,
        days: int,
        end_date: datetime | pd.Timestamp | None = None,
        fast_mode: bool = True,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        futures = {}

        with ThreadPoolExecutor(max_workers=min(6, len(NEWS_PROVIDERS))) as pool:
            for provider in NEWS_PROVIDERS:
                if provider == "newsapi":
                    if not self.api_key:
                        self._record_provider(provider, 0, "skipped", "missing NEWS_API_KEY")
                        continue
                    fut = pool.submit(self._fetch_newsapi, query, days, end_date, fast_mode)
                elif provider == "gnews":
                    if not self.gnews_key:
                        self._record_provider(provider, 0, "skipped", "missing GNEWS_API_KEY")
                        continue
                    fut = pool.submit(self._fetch_gnews, query, days, end_date)
                elif provider == "google_news":
                    fut = pool.submit(self._fetch_google_news_rss, query, days)
                else:
                    fut = pool.submit(self._fetch_rss_provider, provider)
                futures[fut] = provider

            for fut in as_completed(futures):
                provider = futures[fut]
                try:
                    df = fut.result()
                    rows = 0 if df is None else len(df)
                    if df is not None and not df.empty:
                        df["provider"] = provider
                        frames.append(df)
                    self._record_provider(provider, rows, "ok" if rows else "empty", "")
                except Exception as exc:
                    self._record_provider(provider, 0, "failed", str(exc))
                    log.warning("News provider %s failed: %s", provider, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        before = len(combined)
        combined = self._dedupe_articles(combined)
        self.duplicates_removed = before - len(combined)
        self.last_source = "multi"
        log.info("Merged %d articles (%d duplicates removed)", len(combined), self.duplicates_removed)
        return combined

    def _record_provider(self, provider: str, rows: int, status: str, error: str) -> None:
        self.provider_stats[provider] = {
            "articles": int(rows),
            "status": status,
            "error": error[:160] if error else "",
        }

    def _fetch_newsapi(
        self,
        query: str,
        days: int,
        end_date: datetime | pd.Timestamp | None = None,
        fast_mode: bool = True,
    ) -> pd.DataFrame:
        now_utc = datetime.now(tz=timezone.utc)
        end_dt = pd.Timestamp(end_date).to_pydatetime() if end_date is not None else now_utc
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
            end_dt = end_dt.astimezone(timezone.utc)

        start_dt = max(end_dt - timedelta(days=days), now_utc - timedelta(days=28))
        if start_dt > end_dt:
            return pd.DataFrame()

        base_params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(NEWSAPI_PAGE_SIZE, 20 if fast_mode else 10),
            "apiKey": self.api_key,
            "page": 1,
        }

        records: List[dict] = []
        current = start_dt.date()
        last = end_dt.date()
        chunk_days = NEWSAPI_FAST_CHUNK_DAYS if fast_mode else 1

        while current <= last and len(records) < NEWSAPI_MAX_ARTICLES:
            chunk_end = min(current + timedelta(days=chunk_days - 1), last)
            params = {
                **base_params,
                "from": current.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
            }
            try:
                data = self._newsapi_request(params)
            except Exception as exc:
                log.warning("NewsAPI window %s to %s failed: %s", current, chunk_end, exc)
                if "429" in str(exc):
                    break
                current = chunk_end + timedelta(days=1)
                continue

            if not isinstance(data, dict) or data.get("status") != "ok":
                log.warning("NewsAPI status=%s: %s", data.get("status"), data.get("message", ""))
                current = chunk_end + timedelta(days=1)
                continue

            for article in data.get("articles") or []:
                if not isinstance(article, dict):
                    continue
                records.append({
                    "title": str(article.get("title") or ""),
                    "description": str(article.get("description") or ""),
                    "source": str((article.get("source") or {}).get("name") or "NewsAPI"),
                    "published_at": str(article.get("publishedAt") or ""),
                    "url": str(article.get("url") or ""),
                })
                if len(records) >= NEWSAPI_MAX_ARTICLES:
                    break

            current = chunk_end + timedelta(days=1)
            time.sleep(0.05 if fast_mode else 0.1)

        return pd.DataFrame(records)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(NEWSAPI_RETRY_TIMES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(log, 20),
        reraise=True,
    )
    def _newsapi_request(self, params: dict) -> dict:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=NEWSAPI_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _fetch_gnews(
        self,
        query: str,
        days: int,
        end_date: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        end_dt = pd.Timestamp(end_date or datetime.now(tz=timezone.utc))
        start_dt = end_dt - pd.Timedelta(days=days)
        params = {
            "q": query,
            "lang": "en",
            "max": 100,
            "from": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "apikey": self.gnews_key,
        }
        resp = requests.get(GNEWS_BASE, params=params, timeout=NEWSAPI_TIMEOUT)
        resp.raise_for_status()
        records = []
        for article in (resp.json().get("articles") or []):
            records.append({
                "title": str(article.get("title") or ""),
                "description": str(article.get("description") or ""),
                "source": str((article.get("source") or {}).get("name") or "GNews"),
                "published_at": str(article.get("publishedAt") or ""),
                "url": str(article.get("url") or ""),
            })
        return pd.DataFrame(records)

    def _fetch_google_news_rss(self, query: str, days: int) -> pd.DataFrame:
        rss_query = quote_plus(f"{query} when:{max(1, min(days, 30))}d")
        url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"
        return self._fetch_rss_urls("google_news", [url])

    def _fetch_rss_provider(self, provider: str) -> pd.DataFrame:
        return self._fetch_rss_urls(provider, RSS_PROVIDER_URLS.get(provider, []))

    def _fetch_rss_urls(self, provider: str, urls: list[str]) -> pd.DataFrame:
        records: List[dict] = []
        for url in urls:
            try:
                feed = feedparser.parse(url, request_headers=_HEADERS)
                for entry in feed.entries or []:
                    records.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "source": feed.feed.get("title", provider),
                        "published_at": entry.get("published", datetime.now(tz=timezone.utc).isoformat()),
                        "url": entry.get("link", ""),
                    })
                log.info("%s fetched %d articles from %s", provider, len(feed.entries or []), url)
            except Exception as exc:
                log.warning("%s fetch failed for %s: %s", provider, url, exc)
        return pd.DataFrame(records)

    def _dedupe_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        for col in ("title", "url", "source", "published_at"):
            if col not in df.columns:
                df[col] = ""

        parsed_dates = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["_url_key"] = df["url"].fillna("").str.lower().str.strip().str.rstrip("/")
        df["_title_key"] = (
            df["title"].fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["_date_key"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("")
        df["_source_key"] = df["source"].fillna("").str.lower().str.strip()

        with_url = df[df["_url_key"] != ""].drop_duplicates(subset=["_url_key"], keep="first")
        without_url = df[df["_url_key"] == ""]
        deduped = pd.concat([with_url, without_url], ignore_index=True)
        deduped = deduped.drop_duplicates(
            subset=["_title_key", "_date_key", "_source_key"],
            keep="first",
        )
        return deduped.drop(columns=["_url_key", "_title_key", "_date_key", "_source_key"])

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        for col in ("title", "description", "source", "published_at", "url", "provider"):
            if col not in df.columns:
                df[col] = ""

        df = df.copy()

        def _scrub(text: str) -> str:
            text = re.sub(HTML_TAG_PATTERN, " ", str(text))
            text = re.sub(URL_PATTERN, " ", text)
            return " ".join(text.split()).strip()

        df["title"] = df["title"].apply(_scrub)
        df["description"] = df["description"].apply(_scrub)
        df = df[df["title"].str.len() >= MIN_TEXT_LEN].copy()

        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"])
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
        df["text"] = (df["title"].fillna("") + ". " + df["description"].fillna("")).str.strip(". ")
        df["published_at"] = df["published_at"].dt.tz_convert("UTC")

        return df[["title", "description", "text", "source", "provider", "published_at", "url"]]


if __name__ == "__main__":
    fetcher = NewsFetcher(use_cache=False)
    out = fetcher.fetch(query="S&P 500 stock market", days=7)
    print(out[["title", "source", "provider", "published_at"]].head(10).to_string())
    print(fetcher.provider_stats)
