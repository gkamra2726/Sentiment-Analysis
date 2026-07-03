"""
sentiment_scorer.py
--------------------
Sentiment analysis with VADER (primary) + TextBlob (secondary).
Includes text preprocessing, word-frequency analysis, and
keyword extraction.

Public API
----------
    scorer = SentimentScorer()
    df     = scorer.score(articles_df)
    daily  = scorer.aggregate_daily(df)
    kw     = scorer.top_keywords(articles_df, n=20)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    HTML_TAG_PATTERN,
    MIN_TEXT_LEN,
    PUNCT_EXTRA,
    SENTIMENT_MA_WINDOW,
    URL_PATTERN,
    VADER_NEG_THRESHOLD,
    VADER_POS_THRESHOLD,
)
from logger import get_logger

log = get_logger(__name__)

# Common finance / filler stop-words (supplement NLTK if unavailable)
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "its", "it", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "not",
    "as", "up", "out", "s", "new", "says", "say", "said", "after", "before",
    "over", "under", "about", "into", "through", "during", "than", "more",
    "also", "amid", "amid", "within", "amid", "amid", "per", "amid", "amid",
}


class SentimentScorer:
    """
    Multi-engine sentiment scorer for financial news headlines.

    Engines
    -------
    • VADER  — rule-based, fast, finance-aware lexicon
    • TextBlob — statistical polarity/subjectivity (secondary signal)

    Parameters
    ----------
    engine : {'vader', 'vader+textblob'}
        Which engines to run. Default is both.
    """

    def __init__(self, engine: str = "vader+textblob"):
        self.engine   = engine
        self._vader   = SentimentIntensityAnalyzer()
        log.debug("SentimentScorer initialised (engine=%s)", engine)

    # ─────────────────────────────────────────────────────────────
    # Article-level scoring
    # ─────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment columns to the article DataFrame.

        Input columns needed: text (or title), published_at

        Added columns
        -------------
        compound      VADER compound  [-1, +1]
        vader_pos     VADER pos component
        vader_neg     VADER neg component
        vader_neu     VADER neu component
        sentiment     label: 'positive' | 'neutral' | 'negative'
        textblob_pol  TextBlob polarity  [-1, +1]  (if engine includes textblob)
        textblob_sub  TextBlob subjectivity  [0, 1]
        date          calendar date of published_at
        """
        if df.empty:
            return df

        df = df.copy()
        text_col = "text" if "text" in df.columns else "title"
        texts = df[text_col].fillna("").apply(self._preprocess)

        # VADER
        vader_scores = texts.apply(self._vader.polarity_scores)
        df["compound"]  = vader_scores.apply(lambda s: round(s["compound"], 4))
        df["vader_pos"] = vader_scores.apply(lambda s: round(s["pos"], 4))
        df["vader_neg"] = vader_scores.apply(lambda s: round(s["neg"], 4))
        df["vader_neu"] = vader_scores.apply(lambda s: round(s["neu"], 4))
        df["sentiment"] = df["compound"].apply(self._label)

        # TextBlob (optional but cheap)
        if "textblob" in self.engine:
            tb = texts.apply(lambda t: TextBlob(t).sentiment)
            df["textblob_pol"] = tb.apply(lambda s: round(s.polarity, 4))
            df["textblob_sub"] = tb.apply(lambda s: round(s.subjectivity, 4))

        # Date extraction for daily grouping
        if "published_at" in df.columns:
            df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.date
        else:
            df["date"] = pd.Timestamp.today().date()

        log.info("Scored %d articles", len(df))
        return df

    def _preprocess(self, text: str) -> str:
        """Light-touch cleaning before passing to sentiment engines."""
        text = re.sub(HTML_TAG_PATTERN, " ", str(text))
        text = re.sub(URL_PATTERN, " ", text)
        # Collapse whitespace
        return " ".join(text.split())

    def _label(self, score: float) -> str:
        if score >= VADER_POS_THRESHOLD:
            return "positive"
        if score <= VADER_NEG_THRESHOLD:
            return "negative"
        return "neutral"

    # ─────────────────────────────────────────────────────────────
    # Daily aggregation
    # ─────────────────────────────────────────────────────────────

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate article scores to daily metrics.

        Returns
        -------
        pd.DataFrame indexed by date, columns:
            compound_mean, compound_std, compound_median,
            article_count, pct_positive, pct_negative, pct_neutral,
            sentiment_ma5, sentiment_momentum,
            textblob_pol_mean (if available)
        """
        if df.empty or "date" not in df.columns:
            log.warning("aggregate_daily: empty or missing 'date' column")
            return pd.DataFrame()

        grp = df.groupby("date")

        daily = pd.DataFrame({
            "compound_mean":   grp["compound"].mean().round(4),
            "compound_std":    grp["compound"].std().fillna(0).round(4),
            "compound_median": grp["compound"].median().round(4),
            "article_count":   grp["compound"].count(),
            "pct_positive":    grp["sentiment"].apply(
                lambda s: round((s == "positive").sum() / max(len(s), 1) * 100, 1)
            ),
            "pct_negative":    grp["sentiment"].apply(
                lambda s: round((s == "negative").sum() / max(len(s), 1) * 100, 1)
            ),
            "pct_neutral":     grp["sentiment"].apply(
                lambda s: round((s == "neutral").sum() / max(len(s), 1) * 100, 1)
            ),
        })

        if "textblob_pol" in df.columns:
            daily["textblob_pol_mean"] = grp["textblob_pol"].mean().round(4)

        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()

        # Rolling MA
        daily["sentiment_ma5"] = (
            daily["compound_mean"]
            .rolling(window=SENTIMENT_MA_WINDOW, min_periods=1)
            .mean()
            .round(4)
        )

        # Sentiment momentum: difference of 3-day MA minus 7-day MA
        ma3 = daily["compound_mean"].rolling(window=3, min_periods=1).mean()
        ma7 = daily["compound_mean"].rolling(window=7, min_periods=1).mean()
        daily["sentiment_momentum"] = (ma3 - ma7).round(4)

        return daily

    # ─────────────────────────────────────────────────────────────
    # Word frequency & keyword extraction
    # ─────────────────────────────────────────────────────────────

    def top_keywords(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        Return the top-n most frequent content words across all articles.

        Parameters
        ----------
        df : DataFrame with 'text' column
        n  : number of keywords to return

        Returns
        -------
        pd.DataFrame with columns: word, count, frequency_pct
        """
        if df.empty or "text" not in df.columns:
            return pd.DataFrame(columns=["word", "count", "frequency_pct"])

        all_words: List[str] = []
        for text in df["text"].fillna(""):
            text = re.sub(HTML_TAG_PATTERN, " ", str(text))
            text = re.sub(URL_PATTERN, " ", text)
            text = text.lower()
            # Keep only alpha tokens
            words = re.findall(r"\b[a-z]{3,}\b", text)
            all_words.extend(w for w in words if w not in _STOP_WORDS)

        if not all_words:
            return pd.DataFrame(columns=["word", "count", "frequency_pct"])

        total = len(all_words)
        counter = Counter(all_words)
        top = counter.most_common(n)
        return pd.DataFrame([
            {"word": w, "count": c, "frequency_pct": round(c / total * 100, 3)}
            for w, c in top
        ])

    def word_frequency(self, df: pd.DataFrame) -> Dict[str, int]:
        """Return full word-count dict for word-cloud generation."""
        kw = self.top_keywords(df, n=200)
        return dict(zip(kw["word"], kw["count"]))

    # ─────────────────────────────────────────────────────────────
    # Summary stats
    # ─────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> dict:
        """High-level stats dict for dashboard / CLI display."""
        if df.empty:
            return {}

        total = len(df)
        result: dict = {
            "total_articles": total,
            "avg_compound":   round(float(df["compound"].mean()), 4),
            "pct_positive":   round((df["sentiment"] == "positive").sum() / total * 100, 1),
            "pct_negative":   round((df["sentiment"] == "negative").sum() / total * 100, 1),
            "pct_neutral":    round((df["sentiment"] == "neutral").sum()  / total * 100, 1),
        }

        # Most/least positive headlines (safe even if single article)
        if not df.empty and "compound" in df.columns:
            result["most_positive"] = df.loc[df["compound"].idxmax(), "title"]
            result["most_negative"] = df.loc[df["compound"].idxmin(), "title"]

        if "textblob_pol" in df.columns:
            result["avg_textblob_polarity"] = round(float(df["textblob_pol"].mean()), 4)

        return result


# ─────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame({
        "title": [
            "Apple stock surges to record high on blowout earnings",
            "Tech sector faces regulatory scrutiny over AI concerns",
            "Markets mixed ahead of Federal Reserve interest rate decision",
            "Nvidia beats expectations, shares rally 8% in after-hours",
            "Investors cautious amid rising inflation data",
        ],
        "text": [
            "Apple stock surges to record high on blowout earnings. Strong iPhone sales.",
            "Tech sector faces regulatory scrutiny over AI concerns. New framework proposed.",
            "Markets mixed ahead of Federal Reserve interest rate decision. Analysts divided.",
            "Nvidia beats expectations, shares rally 8% after-hours. Data center demand soars.",
            "Investors cautious amid rising inflation data. CPI above forecasts.",
        ],
        "published_at": pd.date_range("2026-06-01", periods=5, freq="D", tz="UTC"),
    })

    scorer = SentimentScorer()
    scored = scorer.score(sample)
    print(scored[["title", "compound", "textblob_pol", "sentiment"]].to_string())
    print("\nDaily:")
    print(scorer.aggregate_daily(scored).to_string())
    print("\nKeywords:")
    print(scorer.top_keywords(scored, n=10).to_string())
