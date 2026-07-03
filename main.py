"""
main.py
-------
Entry point for the Market Sentiment Analyzer.

Pipeline
--------
1. Fetch news headlines (NewsAPI → RSS fallback)
2. Score each article with VADER + TextBlob
3. Aggregate to daily sentiment metrics
4. Download market price data (yfinance)
5. Align and merge on date
6. Run full correlation analysis
7. Generate interactive Plotly charts
8. Save results to CSV

Usage
-----
    python main.py                         # default: S&P 500, tech, 30 days
    python main.py --sector finance        # financials query
    python main.py --days 14              # 2-week window
    python main.py --ticker QQQ           # Nasdaq ETF
    python main.py --no-plots             # skip chart generation
    python main.py --offline              # use only cached data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src/ to sys.path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.news_fetcher       import NewsFetcher
from src.sentiment_scorer   import SentimentScorer
from src.market_data        import MarketData
from src.correlation_engine import CorrelationEngine
from src.visualizer         import Visualizer
from src.config             import OUTPUT_DIR, SECTOR_QUERIES
from src.logger             import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    sector:     str  = "tech",
    days:       int  = 30,
    ticker:     str  = "^GSPC",
    plots:      bool = True,
    output_dir: str  = str(OUTPUT_DIR),
    offline:    bool = False,
) -> dict:
    """
    Execute the full sentiment analysis pipeline.

    Returns dict with all intermediate and final DataFrames plus results.
    On non-critical failures the pipeline continues with whatever data
    is available rather than terminating.
    """
    log.info("=" * 60)
    log.info("  MARKET SENTIMENT ANALYZER")
    log.info("=" * 60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    query = SECTOR_QUERIES.get(sector, SECTOR_QUERIES["all"])

    # ── 1. Fetch news ─────────────────────────────────────────────
    log.info("[1/5] Fetching headlines — sector: %s", sector.upper())
    log.info("      Query: '%s'", query)

    try:
        fetcher     = NewsFetcher(use_cache=not offline)
        articles_df = fetcher.fetch(query=query, days=days)
    except Exception as exc:
        log.error("News fetch failed: %s", exc)
        articles_df = pd.DataFrame()

    if articles_df.empty:
        log.warning("No articles fetched. Sentiment will be empty.")

    # ── 2. Score sentiment ────────────────────────────────────────
    log.info("[2/5] Scoring %d articles", len(articles_df))

    scorer          = SentimentScorer()
    articles_df     = scorer.score(articles_df) if not articles_df.empty else articles_df
    daily_sentiment = scorer.aggregate_daily(articles_df) if not articles_df.empty else pd.DataFrame()

    if articles_df.empty:
        log.warning("Sentiment scoring skipped — no articles available.")
    else:
        summary = scorer.summary(articles_df)
        log.info(
            "Avg compound: %.4f  |  Pos: %.1f%%  Neu: %.1f%%  Neg: %.1f%%",
            summary.get("avg_compound", 0),
            summary.get("pct_positive", 0),
            summary.get("pct_neutral", 0),
            summary.get("pct_negative", 0),
        )

    # ── 3. Market data ────────────────────────────────────────────
    log.info("[3/5] Downloading %s price data (%d days)", ticker, days)

    try:
        md        = MarketData(ticker=ticker, use_cache=not offline)
        market_df = md.fetch(days=days)
    except Exception as exc:
        log.error("Market data download failed: %s", exc)
        market_df = pd.DataFrame()

    if market_df.empty:
        log.warning("No market data available. Analysis cannot proceed.")
        return _empty_return(articles_df, daily_sentiment)

    # ── 4. Align ──────────────────────────────────────────────────
    log.info("[4/5] Aligning sentiment and price data")

    if daily_sentiment.empty:
        log.warning("Daily sentiment is empty — creating synthetic neutral scores")
        daily_sentiment = _synthetic_sentiment(market_df)

    try:
        merged_df = md.align_with_sentiment(market_df, daily_sentiment)
    except Exception as exc:
        log.error("Data alignment failed: %s", exc)
        merged_df = pd.DataFrame()

    if merged_df.empty or len(merged_df) < 3:
        log.warning("Insufficient overlapping data (%d rows)", len(merged_df))
        return _empty_return(articles_df, daily_sentiment)

    if len(merged_df) < 10:
        log.warning("Only %d overlapping days — results may not be statistically meaningful", len(merged_df))

    # Add majority sentiment label per day
    if not articles_df.empty and {"date", "sentiment"}.issubset(articles_df.columns):
        try:
            mode_sent = (
                articles_df.groupby("date")["sentiment"]
                .agg(lambda s: s.mode().iloc[0] if not s.empty else "neutral")
                .reset_index()
            )
            mode_sent["date"] = pd.to_datetime(mode_sent["date"], utc=True).dt.normalize()
            merged_df = merged_df.merge(mode_sent, on="date", how="left")
        except Exception as exc:
            log.warning("Could not attach daily sentiment label: %s", exc)

    # ── 5. Correlation analysis ───────────────────────────────────
    log.info("[5/5] Running correlation analysis")

    try:
        engine  = CorrelationEngine(merged_df)
        results = engine.run_all()
        engine.print_summary(results)
    except Exception as exc:
        log.error("Correlation analysis failed: %s", exc)
        results = {}

    # ── Visualization ─────────────────────────────────────────────
    if plots and results:
        log.info("[+] Generating interactive charts")
        viz = Visualizer(output_dir=out)
        try:
            viz.dashboard(merged_df, articles_df, results)
            viz.sentiment_vs_spx(merged_df)
            viz.lagged_correlation(results.get("lagged", pd.DataFrame()))
            viz.rolling_correlation(results.get("rolling", pd.DataFrame()))
            viz.sentiment_distribution(articles_df)
            viz.regression_scatter(merged_df, results.get("regression", {}))
            viz.correlation_heatmap(merged_df)
            viz.source_distribution(articles_df)
            viz.monthly_sentiment_trend(articles_df)
            viz.candlestick_with_sentiment(merged_df)
        except Exception as exc:
            log.error("Visualization error: %s", exc)

    # ── Save CSVs ─────────────────────────────────────────────────
    try:
        articles_df.to_csv(out / "articles_scored.csv", index=False)
        daily_sentiment.to_csv(out / "daily_sentiment.csv")
        merged_df.to_csv(out / "merged_analysis.csv", index=False)
        log.info("CSVs saved to '%s/'", out)
    except Exception as exc:
        log.error("Failed to save CSVs: %s", exc)

    # ── Final summary ─────────────────────────────────────────────
    _print_summary(articles_df, merged_df, results, ticker, sector)

    return {
        "articles_df":      articles_df,
        "daily_sentiment":  daily_sentiment,
        "market_df":        market_df,
        "merged_df":        merged_df,
        "results":          results,
    }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _synthetic_sentiment(market_df: pd.DataFrame) -> pd.DataFrame:
    """Return a flat-neutral sentiment DataFrame aligned to market dates."""
    if market_df.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(market_df["date"], utc=True)
    idx   = pd.DatetimeIndex(dates).normalize()
    return pd.DataFrame(
        {"compound_mean": 0.0, "sentiment_ma5": 0.0, "article_count": 0},
        index=idx,
    )


def _empty_return(articles_df, daily_sentiment) -> dict:
    return {
        "articles_df":     articles_df,
        "daily_sentiment": daily_sentiment,
        "market_df":       pd.DataFrame(),
        "merged_df":       pd.DataFrame(),
        "results":         {},
    }


def _print_summary(articles_df, merged_df, results, ticker, sector):
    p   = results.get("pearson", {})
    b   = results.get("best_lag", {})
    reg = results.get("regression", {})
    spx_ret = (
        merged_df["cumulative_return"].iloc[-1] * 100
        if not merged_df.empty and "cumulative_return" in merged_df.columns
        else float("nan")
    )
    lines = [
        "\n" + "=" * 60,
        "  RESULTS SUMMARY",
        "=" * 60,
        f"  Sector              : {sector.upper()}",
        f"  Ticker              : {ticker}",
        f"  Articles analyzed   : {len(articles_df)}",
        f"  Trading days        : {len(merged_df)}",
        f"  Pearson r           : {p.get('r', 'N/A')}  ({p.get('interpretation', '')})",
        f"  p-value             : {p.get('p_value', 'N/A')}"
        f"  {'✓ significant' if p.get('significant') else '✗ not significant'}",
        f"  95% CI              : [{p.get('ci_lower', 'N/A')}, {p.get('ci_upper', 'N/A')}]",
        f"  R²                  : {reg.get('r_squared', 'N/A')}",
        f"  Best lag            : {b.get('lag_days', 'N/A')} day(s)",
        f"  {ticker} period return : {spx_ret:.2f}%",
        "=" * 60,
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Market Sentiment Analyzer — NewsAPI + VADER + S&P 500 correlation"
    )
    parser.add_argument("--sector", default="tech",
                        choices=list(SECTOR_QUERIES.keys()),
                        help="News sector to analyze (default: tech)")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to analyze (default: 30)")
    parser.add_argument("--ticker", default="^GSPC",
                        help="Yahoo Finance ticker (default: ^GSPC)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory (default: outputs/)")
    parser.add_argument("--offline", action="store_true",
                        help="Use only cached data (no network requests)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        sector=args.sector,
        days=args.days,
        ticker=args.ticker,
        plots=not args.no_plots,
        output_dir=args.output_dir,
        offline=args.offline,
    )
