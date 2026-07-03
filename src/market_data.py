"""
market_data.py
--------------
Downloads OHLCV price data via Stooq, with yfinance as a fallback:
  - Robust error handling (no crash on empty/bad data)
  - Timezone normalisation (UTC)
  - File-based caching
  - Configurable ticker and date range

Public API
----------
    md  = MarketData(ticker="^GSPC")
    df  = md.fetch(days=30)
    merged = md.align_with_sentiment(df, daily_sentiment)
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from cache import cache_get, cache_set
from config import (
    ALPHA_VANTAGE_BASE,
    ALPHA_VANTAGE_KEY,
    ALPHA_VANTAGE_SYMBOL_MAP,
    DEFAULT_TICKER,
    FRED_DOWNLOAD_URL,
    FRED_SYMBOL_MAP,
    MARKET_CACHE_TTL_HOURS,
    MARKET_DATA_PROVIDER,
    MARKET_DATE_BUFFER,
    MARKET_PROVIDER_PRIORITY,
    MARKET_TIMEOUT,
    STOOQ_DOWNLOAD_URL,
    STOOQ_SYMBOL_MAP,
)
from logger import get_logger

log = get_logger(__name__)


class MarketData:
    """
    Downloads and processes OHLCV market data.

    Parameters
    ----------
    ticker    : Market symbol (default '^GSPC' = S&P 500)
    use_cache : bool  (default True)
    """

    def __init__(self, ticker: str = DEFAULT_TICKER, use_cache: bool = True):
        self.ticker    = ticker
        self.use_cache = use_cache
        self.last_provider = ""
        self.provider_status: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────

    def fetch(self, days: int = 30) -> pd.DataFrame:
        """
        Download price data for the past *days* calendar days.

        Returns
        -------
        pd.DataFrame or empty DataFrame on failure.
        Columns: date, open, high, low, close, volume,
                 daily_return, cumulative_return, normalized_price
        """
        cache_key = f"market::{self.ticker}::days{days}"
        if self.use_cache:
            cached = cache_get(cache_key)
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                log.info("Market data loaded from cache (%d rows)", len(cached))
                return cached

        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days + MARKET_DATE_BUFFER)

        raw = self._download(start, end)

        if raw is None or raw.empty:
            log.warning("No data returned for ticker %s", self.ticker)
            return pd.DataFrame()

        df = self._process(raw)
        df = df.tail(days).reset_index(drop=True)
        log.info("Got %d trading days for %s", len(df), self.ticker)

        if self.use_cache:
            cache_set(cache_key, df, ttl_hours=MARKET_CACHE_TTL_HOURS)

        return df

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        """Fetch by explicit date strings ('YYYY-MM-DD')."""
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        raw = self._download(start_dt, end_dt)

        if raw is None or raw.empty:
            return pd.DataFrame()

        return self._process(raw).reset_index(drop=True)

    def align_with_sentiment(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join market data and daily sentiment scores on date.

        Parameters
        ----------
        market_df     : output of fetch()
        sentiment_df  : output of SentimentScorer.aggregate_daily()

        Returns
        -------
        Merged DataFrame with both market metrics and sentiment columns.
        Empty DataFrame if either input is empty.
        """
        if market_df.empty:
            log.warning("align_with_sentiment: market_df is empty")
            return pd.DataFrame()
        if sentiment_df.empty:
            log.warning("align_with_sentiment: sentiment_df is empty")
            return pd.DataFrame()

        m = market_df.copy()
        s = sentiment_df.copy()

        # Normalise to UTC date (no time component)
        m["date"] = pd.to_datetime(m["date"], utc=True, errors="coerce").dt.normalize()
        s.index   = pd.to_datetime(s.index, utc=True, errors="coerce").normalize()

        merged = m.set_index("date").join(s, how="inner")
        merged = merged.dropna(subset=["daily_return", "compound_mean"])
        merged = merged.reset_index()

        log.info("Aligned %d overlapping trading days", len(merged))
        return merged

    # ─────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────

    def _download(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Download with the configured provider first, then try the fallback."""
        valid = {"fred", "alpha_vantage", "stooq", "yfinance"}
        provider = MARKET_DATA_PROVIDER if MARKET_DATA_PROVIDER in valid else "fred"
        priority = [p for p in [provider] + MARKET_PROVIDER_PRIORITY if p in valid]
        providers = list(dict.fromkeys(priority + ["fred", "alpha_vantage", "stooq", "yfinance"]))

        for name in providers:
            if name == "fred":
                raw = self._download_fred(start, end)
            elif name == "alpha_vantage":
                raw = self._download_alpha_vantage(start, end)
            elif name == "stooq":
                raw = self._download_stooq(start, end)
            else:
                raw = self._download_yfinance(start, end)

            if raw is not None and not raw.empty:
                self.last_provider = name
                self.provider_status[name] = {"status": "ok", "rows": len(raw)}
                return raw
            self.provider_status[name] = {"status": "empty", "rows": 0}

        return pd.DataFrame()

    def _download_alpha_vantage(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Download adjusted daily data from Alpha Vantage when a key is configured."""
        if not ALPHA_VANTAGE_KEY:
            return pd.DataFrame()

        symbol = ALPHA_VANTAGE_SYMBOL_MAP.get(self.ticker.strip().upper(), self.ticker.strip().upper())
        if symbol.startswith("^"):
            return pd.DataFrame()

        log.info("Downloading %s from Alpha Vantage as %s", self.ticker, symbol)
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": ALPHA_VANTAGE_KEY,
        }
        try:
            resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=MARKET_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.warning("Alpha Vantage download failed for %s: %s", self.ticker, exc)
            return pd.DataFrame()

        series = data.get("Time Series (Daily)")
        if not isinstance(series, dict):
            log.warning("Alpha Vantage returned no daily series for %s: %s", symbol, data.get("Note") or data.get("Error Message"))
            return pd.DataFrame()

        rows = []
        for date_str, values in series.items():
            dt = pd.to_datetime(date_str, utc=True, errors="coerce")
            if pd.isna(dt) or dt < start or dt > end:
                continue
            rows.append({
                "date": dt,
                "open": values.get("1. open"),
                "high": values.get("2. high"),
                "low": values.get("3. low"),
                "close": values.get("5. adjusted close") or values.get("4. close"),
                "volume": values.get("6. volume"),
            })

        if not rows:
            return pd.DataFrame()

        raw = pd.DataFrame(rows).set_index("date").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        return raw.dropna(subset=["close"])

    def _download_fred(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Download daily index closes from FRED and shape them like OHLCV data."""
        series_id = FRED_SYMBOL_MAP.get(self.ticker.strip().upper())
        if not series_id:
            return pd.DataFrame()

        log.info("Downloading %s from FRED as %s (%s to %s)",
                 self.ticker, series_id, start.date(), end.date())
        try:
            resp = requests.get(
                FRED_DOWNLOAD_URL,
                params={"id": series_id},
                timeout=MARKET_TIMEOUT,
            )
            resp.raise_for_status()
            raw = pd.read_csv(StringIO(resp.text))
        except (requests.RequestException, pd.errors.ParserError) as exc:
            log.warning("FRED download failed for %s: %s", self.ticker, exc)
            return pd.DataFrame()

        if raw.empty or series_id not in raw.columns:
            return pd.DataFrame()

        raw = raw.rename(columns={"observation_date": "date", series_id: "close"})
        raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"])
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        if raw.empty:
            return pd.DataFrame()

        raw["open"] = raw["close"]
        raw["high"] = raw["close"]
        raw["low"] = raw["close"]
        raw["volume"] = 0
        return raw.set_index("date")[["open", "high", "low", "close", "volume"]]

    def _download_yfinance(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Download via Yahoo/yfinance. Kept as fallback because Yahoo may rate-limit."""
        log.info("Downloading %s from Yahoo Finance (%s to %s)",
                 self.ticker, start.date(), end.date())
        try:
            return yf.download(
                self.ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                timeout=MARKET_TIMEOUT,
            )
        except Exception as exc:
            log.warning("Yahoo Finance download failed for %s: %s", self.ticker, exc)
            return pd.DataFrame()

    def _download_stooq(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Download daily OHLCV data from Stooq's CSV endpoint."""
        symbol = self._stooq_symbol(self.ticker)
        params = {
            "s": symbol,
            "i": "d",
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
        }
        log.info("Downloading %s from Stooq as %s (%s to %s)",
                 self.ticker, symbol, start.date(), end.date())

        try:
            resp = self._stooq_get(params)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Stooq download failed for %s: %s", self.ticker, exc)
            return pd.DataFrame()

        text = resp.text.strip()
        if not text or text.lower().startswith("no data"):
            log.warning("Stooq returned no data for %s (%s)", self.ticker, symbol)
            return pd.DataFrame()

        try:
            raw = pd.read_csv(StringIO(text))
        except Exception as exc:
            log.warning("Could not parse Stooq response for %s: %s", self.ticker, exc)
            return pd.DataFrame()

        if raw.empty or "Date" not in raw.columns:
            log.warning("Stooq response missing OHLC data for %s", self.ticker)
            return pd.DataFrame()

        raw = raw.rename(columns={"Date": "date"})
        raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["date"]).set_index("date")
        return raw.rename(columns=str.lower)

    def _stooq_get(self, params: dict[str, str]) -> requests.Response:
        """Fetch Stooq CSV and pass its lightweight JS verification when present."""
        session = requests.Session()
        resp = session.get(STOOQ_DOWNLOAD_URL, params=params, timeout=MARKET_TIMEOUT)
        if "__verify" not in resp.text:
            return resp

        challenge = re.search(r'const c="([^"]+)",d=(\d+)', resp.text)
        if not challenge:
            return resp

        token, difficulty_text = challenge.groups()
        difficulty = int(difficulty_text)
        prefix = "0" * difficulty
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{token}{nonce}".encode()).hexdigest()
            if digest.startswith(prefix):
                break
            nonce += 1

        verify_url = "https://stooq.com/__verify"
        session.post(
            verify_url,
            data={"c": token, "n": str(nonce)},
            timeout=MARKET_TIMEOUT,
        ).raise_for_status()
        return session.get(STOOQ_DOWNLOAD_URL, params=params, timeout=MARKET_TIMEOUT)

    @staticmethod
    def _stooq_symbol(ticker: str) -> str:
        """Convert common Yahoo symbols to Stooq symbols."""
        cleaned = ticker.strip()
        mapped = STOOQ_SYMBOL_MAP.get(cleaned.upper())
        if mapped:
            return mapped
        if cleaned.startswith("^"):
            return cleaned.lower()
        if "." in cleaned:
            return cleaned.lower()
        return f"{cleaned.lower()}.us"

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-index yfinance output and compute metrics."""
        # yfinance ≥ 0.2 returns a MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        needed = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
        df = raw[needed].copy()
        df.index.name = "date"
        df = df.reset_index()

        # Normalise to UTC-aware date
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "close"])

        # Return metrics
        df["daily_return"] = df["close"].pct_change().round(6)
        df["cumulative_return"] = (df["close"] / df["close"].iloc[0] - 1).round(6)

        # Normalise price to [-1, +1] (same scale as VADER)
        mn, mx = df["close"].min(), df["close"].max()
        if mx > mn:
            df["normalized_price"] = ((df["close"] - mn) / (mx - mn) * 2 - 1).round(4)
        else:
            df["normalized_price"] = 0.0

        # Volatility (rolling 5-day std of daily returns)
        df["volatility_5d"] = df["daily_return"].rolling(5, min_periods=1).std().round(6)

        return df


# ─────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    md = MarketData(use_cache=False)
    df = md.fetch(days=30)
    if not df.empty:
        print(df[["date", "close", "daily_return", "cumulative_return"]].to_string())
        print(f"\nSPX 30d return: {df['cumulative_return'].iloc[-1]*100:.2f}%")
    else:
        print("No data returned (possibly offline)")
