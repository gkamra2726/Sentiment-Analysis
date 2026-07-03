"""
tests.py
--------
Comprehensive unit tests for Market Sentiment Analyzer v2.
Runs entirely offline — no API calls required.

Run:
    python tests.py
    python -m pytest tests.py -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.sentiment_scorer   import SentimentScorer
from src.market_data        import MarketData
from src.correlation_engine import CorrelationEngine
from src.cache              import cache_get, cache_set, _key_path
from src.news_fetcher       import NewsFetcher


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def make_articles(n: int = 20) -> pd.DataFrame:
    headlines = [
        "Apple stock surges to record high on blowout earnings",
        "Tech sector faces regulatory scrutiny over AI concerns",
        "Markets mixed ahead of Federal Reserve interest rate decision",
        "Nvidia beats expectations, shares rally 8% in after-hours",
        "Investors cautious amid rising inflation data",
        "Amazon revenue growth disappoints Wall Street analysts",
        "Strong job report fuels optimism about soft landing",
        "Oil prices plunge on demand concern from China slowdown",
        "Meta reports record ad revenue, stock up 6%",
        "Crypto market crashes on regulatory crackdown fears",
    ] * (n // 10 + 1)

    return pd.DataFrame({
        "title":        headlines[:n],
        "description":  ["Details: " + h for h in headlines[:n]],
        "text":         headlines[:n],
        "source":       ["Reuters", "Bloomberg", "CNBC", "WSJ", "FT"] * (n // 5 + 1),
        "published_at": pd.date_range("2026-06-01", periods=n, freq="D", tz="UTC"),
        "url":          ["https://example.com"] * n,
    })


def make_merged(n: int = 25) -> pd.DataFrame:
    np.random.seed(42)
    s   = np.random.uniform(-0.5, 0.7, n)
    ret = s * 0.003 + np.random.normal(0, 0.005, n)
    c   = 5000 + np.cumsum(ret * 500)
    return pd.DataFrame({
        "date":               pd.date_range("2026-06-01", periods=n, tz="UTC"),
        "sentiment_ma5":      s,
        "compound_mean":      s * 0.9,
        "compound_std":       np.abs(np.random.normal(0.1, 0.04, n)),
        "daily_return":       ret,
        "cumulative_return":  np.cumsum(ret),
        "close":              c,
        "open":               c * 0.998,
        "high":               c * 1.004,
        "low":                c * 0.996,
        "volume":             np.random.randint(int(1e9), int(4e9), n),
        "article_count":      np.random.randint(3, 20, n),
        "pct_positive":       np.random.uniform(30, 70, n),
        "pct_negative":       np.random.uniform(10, 40, n),
        "volatility_5d":      np.abs(np.random.normal(0.005, 0.002, n)),
    })


# ─────────────────────────────────────────────────────────────────
# SentimentScorer
# ─────────────────────────────────────────────────────────────────

class TestSentimentScorer(unittest.TestCase):

    def setUp(self):
        self.scorer   = SentimentScorer()
        self.articles = make_articles(15)

    def test_score_adds_compound(self):
        df = self.scorer.score(self.articles)
        self.assertIn("compound", df.columns)

    def test_compound_range(self):
        df = self.scorer.score(self.articles)
        self.assertTrue((df["compound"] >= -1).all())
        self.assertTrue((df["compound"] <= 1).all())

    def test_sentiment_labels_valid(self):
        df = self.scorer.score(self.articles)
        self.assertTrue(set(df["sentiment"].unique()).issubset({"positive", "neutral", "negative"}))

    def test_textblob_columns_present(self):
        df = self.scorer.score(self.articles)
        self.assertIn("textblob_pol", df.columns)
        self.assertIn("textblob_sub", df.columns)

    def test_positive_headline(self):
        df = self.scorer.score(pd.DataFrame({
            "text": ["Stocks surge to record high on strong earnings beat"],
            "published_at": [datetime.now(tz=timezone.utc)],
        }))
        self.assertGreater(df["compound"].iloc[0], 0)

    def test_negative_headline(self):
        df = self.scorer.score(pd.DataFrame({
            "text": ["Market crashes, stocks plunge on recession fears disaster"],
            "published_at": [datetime.now(tz=timezone.utc)],
        }))
        self.assertLess(df["compound"].iloc[0], 0)

    def test_date_column_extracted(self):
        df = self.scorer.score(self.articles)
        self.assertIn("date", df.columns)

    def test_aggregate_daily_columns(self):
        df    = self.scorer.score(self.articles)
        daily = self.scorer.aggregate_daily(df)
        for col in ("compound_mean", "compound_std", "article_count", "sentiment_ma5"):
            self.assertIn(col, daily.columns)

    def test_aggregate_daily_ma5_not_all_null(self):
        df    = self.scorer.score(self.articles)
        daily = self.scorer.aggregate_daily(df)
        self.assertFalse(daily["sentiment_ma5"].isna().all())

    def test_sentiment_momentum_column(self):
        df    = self.scorer.score(self.articles)
        daily = self.scorer.aggregate_daily(df)
        self.assertIn("sentiment_momentum", daily.columns)

    def test_summary_keys(self):
        df      = self.scorer.score(self.articles)
        summary = self.scorer.summary(df)
        for k in ("total_articles", "avg_compound", "pct_positive", "pct_negative"):
            self.assertIn(k, summary)

    def test_empty_df_returns_empty(self):
        result = self.scorer.score(pd.DataFrame(columns=["text", "published_at"]))
        self.assertTrue(result.empty)

    def test_top_keywords_returns_df(self):
        df = self.scorer.score(self.articles)
        kw = self.scorer.top_keywords(df, n=10)
        self.assertIn("word", kw.columns)
        self.assertIn("count", kw.columns)
        self.assertLessEqual(len(kw), 10)

    def test_word_frequency_is_dict(self):
        df   = self.scorer.score(self.articles)
        freq = self.scorer.word_frequency(df)
        self.assertIsInstance(freq, dict)
        self.assertTrue(all(isinstance(v, int) for v in freq.values()))

    def test_aggregate_empty_returns_empty(self):
        result = self.scorer.aggregate_daily(pd.DataFrame())
        self.assertTrue(result.empty)


# ─────────────────────────────────────────────────────────────────
# MarketData
# ─────────────────────────────────────────────────────────────────

class TestMarketData(unittest.TestCase):

    def setUp(self):
        self.md = MarketData(ticker="^GSPC", use_cache=False)

    def _fake_raw(self, n: int = 20) -> pd.DataFrame:
        dates  = pd.date_range("2026-06-01", periods=n, tz="UTC")
        tuples = [(c, "^GSPC") for c in ["Open", "High", "Low", "Close", "Volume"]]
        cols   = pd.MultiIndex.from_tuples(tuples)
        data   = np.random.uniform(4900, 5200, (n, 5))
        df     = pd.DataFrame(data, index=dates, columns=cols)
        df.index.name = "Date"
        return df

    def test_process_adds_return_cols(self):
        df = self.md._process(self._fake_raw(20))
        self.assertIn("daily_return", df.columns)
        self.assertIn("cumulative_return", df.columns)
        self.assertIn("normalized_price", df.columns)

    def test_normalized_price_range(self):
        df = self.md._process(self._fake_raw(20))
        self.assertGreaterEqual(df["normalized_price"].min(), -1.01)
        self.assertLessEqual(df["normalized_price"].max(), 1.01)

    def test_volatility_column(self):
        df = self.md._process(self._fake_raw(20))
        self.assertIn("volatility_5d", df.columns)

    def test_align_with_sentiment_inner_join(self):
        market = make_merged(15)[["date", "daily_return", "close",
                                  "cumulative_return", "volatility_5d"]].copy()
        daily_sent = pd.DataFrame(
            {"compound_mean": np.random.uniform(-0.5, 0.5, 10),
             "sentiment_ma5": np.random.uniform(-0.5, 0.5, 10)},
            index=pd.date_range("2026-06-03", periods=10, tz="UTC"),
        )
        merged = self.md.align_with_sentiment(market, daily_sent)
        self.assertIn("compound_mean", merged.columns)
        self.assertGreater(len(merged), 0)

    def test_align_with_empty_market_returns_empty(self):
        result = self.md.align_with_sentiment(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(result.empty)

    def test_process_single_row(self):
        """Should not crash on a single-row DataFrame."""
        df = self.md._process(self._fake_raw(1))
        self.assertEqual(len(df), 1)


# ─────────────────────────────────────────────────────────────────
# CorrelationEngine
# ─────────────────────────────────────────────────────────────────

class TestCorrelationEngine(unittest.TestCase):

    def setUp(self):
        self.merged = make_merged(30)
        self.engine = CorrelationEngine(self.merged)

    def test_pearson_keys(self):
        r = self.engine.pearson()
        for k in ("r", "p_value", "significant", "interpretation", "ci_lower", "ci_upper"):
            self.assertIn(k, r)

    def test_pearson_r_range(self):
        r = self.engine.pearson()["r"]
        self.assertGreaterEqual(r, -1)
        self.assertLessEqual(r, 1)

    def test_spearman_keys(self):
        s = self.engine.spearman()
        for k in ("r", "p_value", "significant"):
            self.assertIn(k, s)

    def test_kendall_keys(self):
        k = self.engine.kendall()
        for key in ("tau", "p_value", "significant"):
            self.assertIn(key, k)

    def test_lagged_correlation_df(self):
        df = self.engine.lagged_correlation(max_lag=3)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)   # lags 0, 1, 2, 3

    def test_lagged_no_crash_small_data(self):
        """Should not crash with very few rows."""
        tiny = make_merged(6)
        eng  = CorrelationEngine(tiny)
        df   = eng.lagged_correlation(max_lag=5)
        self.assertIsInstance(df, pd.DataFrame)

    def test_rolling_correlation_df(self):
        df = self.engine.rolling_correlation(window=5)
        self.assertIn("rolling_r", df.columns)

    def test_regression_r2_range(self):
        reg = self.engine.linear_regression()
        self.assertGreaterEqual(reg["r_squared"], 0)
        self.assertLessEqual(reg["r_squared"], 1)

    def test_run_all_keys(self):
        results = self.engine.run_all()
        for k in ("pearson", "spearman", "kendall", "lagged", "rolling",
                  "regression", "best_lag", "volatility"):
            self.assertIn(k, results)

    def test_best_lag_within_max(self):
        results = self.engine.run_all()
        lag = results["best_lag"].get("lag_days", 0)
        self.assertGreaterEqual(lag, 0)
        self.assertLessEqual(lag, 5)

    def test_perfect_correlation(self):
        x  = np.linspace(0, 1, 30)
        df = pd.DataFrame({"sentiment_ma5": x, "daily_return": x * 0.01})
        r  = CorrelationEngine(df).pearson()["r"]
        self.assertAlmostEqual(r, 1.0, places=3)

    def test_empty_df_returns_empty_results(self):
        eng     = CorrelationEngine(pd.DataFrame())
        results = eng.run_all()
        self.assertIn("pearson", results)

    def test_confidence_interval_present(self):
        p = self.engine.pearson()
        self.assertIsNotNone(p.get("ci_lower"))
        self.assertIsNotNone(p.get("ci_upper"))

    def test_volatility_analysis_returns_dict(self):
        v = self.engine.volatility_analysis()
        self.assertIsInstance(v, dict)

    def test_spearman_r_range(self):
        r = self.engine.spearman()["r"]
        self.assertGreaterEqual(r, -1)
        self.assertLessEqual(r, 1)


# ─────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────

class TestCache(unittest.TestCase):

    def test_set_and_get(self):
        cache_set("test_key_abc", {"hello": "world"}, ttl_hours=1)
        result = cache_get("test_key_abc")
        self.assertEqual(result, {"hello": "world"})

    def test_get_missing_key_returns_none(self):
        result = cache_get("definitely_not_a_key_xyz_789")
        self.assertIsNone(result)

    def test_dataframe_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        cache_set("df_test_roundtrip", df, ttl_hours=1)
        recovered = cache_get("df_test_roundtrip")
        self.assertIsNotNone(recovered)
        pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                      recovered.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────
# NewsFetcher (mocked)
# ─────────────────────────────────────────────────────────────────

class TestNewsFetcher(unittest.TestCase):

    def test_clean_removes_html(self):
        fetcher = NewsFetcher(use_cache=False)
        raw = pd.DataFrame({
            "title":        ["<b>Apple</b> stock rises"],
            "description":  ["Prices <a href='x'>rose</a> sharply"],
            "source":       ["Reuters"],
            "published_at": ["2026-06-01T10:00:00Z"],
            "url":          ["https://example.com"],
        })
        cleaned = fetcher._clean(raw)
        self.assertNotIn("<b>", cleaned["title"].iloc[0])

    def test_clean_deduplicates(self):
        fetcher = NewsFetcher(use_cache=False)
        raw = pd.DataFrame({
            "title":        ["Apple rises", "Apple rises"],
            "description":  ["A", "A"],
            "source":       ["Reuters", "Reuters"],
            "published_at": ["2026-06-01T10:00:00Z", "2026-06-01T11:00:00Z"],
            "url":          ["https://a.com", "https://b.com"],
        })
        cleaned = fetcher._clean(raw)
        self.assertEqual(len(cleaned), 1)

    def test_clean_empty_returns_empty(self):
        fetcher = NewsFetcher(use_cache=False)
        result  = fetcher._clean(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_clean_produces_text_column(self):
        fetcher = NewsFetcher(use_cache=False)
        raw = pd.DataFrame({
            "title":        ["Apple beats earnings"],
            "description":  ["Revenue exceeded estimates"],
            "source":       ["Reuters"],
            "published_at": ["2026-06-01T10:00:00Z"],
            "url":          ["https://example.com"],
        })
        cleaned = fetcher._clean(raw)
        self.assertIn("text", cleaned.columns)
        self.assertIn("Apple", cleaned["text"].iloc[0])

    @patch("src.news_fetcher.requests.get")
    def test_newsapi_bad_status_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "error", "message": "API key invalid"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetcher = NewsFetcher(api_key="fake_key", use_cache=False)
        df = fetcher._fetch_newsapi("test", 7)
        self.assertTrue(df.empty)


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite  = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'✓' if result.wasSuccessful() else '✗'} {passed}/{total} tests passed")
