"""
app.py
------
Streamlit dashboard for the Market Sentiment Analyzer.

Run with:
    cd market_sentiment_v2
    streamlit run app.py

Features
--------
• Sidebar: sector selector, date range, ticker, engine toggle
• KPI cards: Pearson r, R², best lag, avg sentiment, article count
• Interactive Plotly charts embedded in Streamlit
• News table with sentiment scores
• Top positive / negative headlines section
• Download buttons for CSV exports
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.news_fetcher       import NewsFetcher
from src.sentiment_scorer   import SentimentScorer
from src.market_data        import MarketData
from src.correlation_engine import CorrelationEngine
from src.visualizer         import Visualizer
from src.config             import MARKET_INDEX_OPTIONS, MIN_OBSERVATIONS, NEWS_CACHE_TTL_HOURS, PALETTE, SECTOR_QUERIES

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.kpi-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 700; }
.kpi-label { font-size: 0.85rem; color: #9095a8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    sector  = st.selectbox("Sector", list(SECTOR_QUERIES.keys()), index=0)
    ticker  = st.selectbox(
        "Market Index",
        list(MARKET_INDEX_OPTIONS.keys()),
        index=0,
        format_func=lambda symbol: f"{MARKET_INDEX_OPTIONS[symbol]} ({symbol})",
        help="Reliable distinct index series from FRED",
    )
    days    = st.slider("Lookback (days)", min_value=7, max_value=90, value=30, step=1)
    fast_mode = st.toggle("Fast mode (fewer NewsAPI calls)", value=True)
    use_cache = st.toggle("Use cached data when available", value=True)

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.caption(f"News results are cached for {NEWS_CACHE_TTL_HOURS} hours to reduce API calls.")

# ─────────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────────
st.title("📊 Market Sentiment Analyzer")
st.caption(
    "Correlates financial news sentiment (VADER + TextBlob) "
    "with S&P 500 / equity index price movements."
)

if not run_btn:
    st.info("Configure settings in the sidebar, then click **Run Analysis**.")
    st.stop()

run_started = time.perf_counter()

# ─────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────
query = SECTOR_QUERIES.get(sector, SECTOR_QUERIES["all"])

with st.spinner("Downloading market data..."):
    md        = MarketData(ticker=ticker, use_cache=use_cache)
    market_df = md.fetch(days=days)

if market_df.empty:
    st.error(f"Could not download market data for **{ticker}**. Try a different ticker.")
    st.stop()

market_end_date = pd.to_datetime(market_df["date"], utc=True).max()

with st.spinner("Fetching news headlines…"):
    fetcher     = NewsFetcher(use_cache=use_cache)
    articles_df = fetcher.fetch(
        query=query,
        days=days,
        end_date=market_end_date,
        fast_mode=fast_mode,
    )

if articles_df.empty:
    st.error("No articles fetched. Check your NEWS_API_KEY or enable offline mode.")
    st.stop()

with st.spinner("Scoring sentiment…"):
    scorer          = SentimentScorer()
    articles_df     = scorer.score(articles_df)
    daily_sentiment = scorer.aggregate_daily(articles_df)
    summary         = scorer.summary(articles_df)
    keywords        = scorer.top_keywords(articles_df, n=20)

with st.spinner("Aligning data and running correlation analysis…"):
    merged_df = md.align_with_sentiment(market_df, daily_sentiment)
    if merged_df.empty:
        st.warning("No overlapping trading days between news and market data.")
        st.stop()
    if len(merged_df) < MIN_OBSERVATIONS:
        st.warning(
            f"Only {len(merged_df)} overlapping trading day(s) found. "
            f"Need at least {MIN_OBSERVATIONS} for reliable correlation charts. "
            "Try using cached data, increasing lookback, or checking the NewsAPI key."
        )
        with st.expander("Debug data coverage", expanded=True):
            st.write("Market data range:", market_df["date"].min(), "to", market_df["date"].max())
            st.write("Daily sentiment range:", daily_sentiment.index.min(), "to", daily_sentiment.index.max())
            st.write("Active market provider:", getattr(md, "last_provider", "unknown") or "cache/unknown")
            st.write("News cache hit:", getattr(fetcher, "cache_hit", False))
            st.write("Duplicate articles removed:", getattr(fetcher, "duplicates_removed", 0))
            provider_stats = getattr(fetcher, "provider_stats", {})
            if provider_stats:
                st.dataframe(pd.DataFrame(provider_stats).T, use_container_width=True)
            st.dataframe(daily_sentiment.tail(10), use_container_width=True)
        st.stop()

    engine  = CorrelationEngine(merged_df)
    results = engine.run_all()

viz = Visualizer(output_dir="outputs")

# ─────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────
pearson  = results.get("pearson", {})
reg      = results.get("regression", {})
best_lag = results.get("best_lag", {})

cols = st.columns(5)
kpis = [
    ("Pearson r",    f"{pearson.get('r', 'N/A')}",          PALETTE["sentiment"]),
    ("R²",           f"{reg.get('r_squared', 'N/A')}",       PALETTE["spx"]),
    ("Best Lag",     f"{best_lag.get('lag_days', 'N/A')}d",  PALETTE["ma"]),
    ("Avg Sentiment",f"{summary.get('avg_compound', 'N/A')}",PALETTE["neutral"]),
    ("Articles",     str(summary.get("total_articles", 0)),  "#9095a8"),
]
for col, (label, value, color) in zip(cols, kpis):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

with st.expander("Operational metrics", expanded=False):
    elapsed = time.perf_counter() - run_started
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Execution time", f"{elapsed:.1f}s")
    m2.metric("News cache", "hit" if getattr(fetcher, "cache_hit", False) else "miss")
    m3.metric("Duplicates removed", getattr(fetcher, "duplicates_removed", 0))
    m4.metric("Market provider", getattr(md, "last_provider", "unknown") or "cache")
    m5.metric("Articles", len(articles_df))

    provider_stats = getattr(fetcher, "provider_stats", {})
    if provider_stats:
        st.markdown("**News provider health**")
        st.dataframe(pd.DataFrame(provider_stats).T, use_container_width=True)

    market_stats = getattr(md, "provider_status", {})
    if market_stats:
        st.markdown("**Market provider health**")
        st.dataframe(pd.DataFrame(market_stats).T, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# Charts (2-col layout)
# ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        viz.sentiment_vs_spx(merged_df, save=False),
        use_container_width=True,
    )

with c2:
    st.plotly_chart(
        viz.candlestick_with_sentiment(merged_df, save=False),
        use_container_width=True,
    )

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        viz.lagged_correlation(results.get("lagged", pd.DataFrame()), save=False),
        use_container_width=True,
    )
with c4:
    st.plotly_chart(
        viz.rolling_correlation(results.get("rolling", pd.DataFrame()), save=False),
        use_container_width=True,
    )

c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(
        viz.regression_scatter(merged_df, reg, save=False),
        use_container_width=True,
    )
with c6:
    st.plotly_chart(
        viz.sentiment_distribution(articles_df, save=False),
        use_container_width=True,
    )

c7, c8 = st.columns(2)
with c7:
    st.plotly_chart(
        viz.source_distribution(articles_df, save=False),
        use_container_width=True,
    )
with c8:
    st.plotly_chart(
        viz.monthly_sentiment_trend(articles_df, save=False),
        use_container_width=True,
    )

st.plotly_chart(
    viz.correlation_heatmap(merged_df, save=False),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────
# Top positive / negative headlines
# ─────────────────────────────────────────────────────────────────
st.subheader("🔝 Top Headlines")
th1, th2 = st.columns(2)

with th1:
    st.markdown("**Most Positive 📈**")
    top_pos = (
        articles_df[articles_df["sentiment"] == "positive"]
        .nlargest(5, "compound")[["title", "compound", "source"]]
    )
    for _, row in top_pos.iterrows():
        st.markdown(f"- **{row['title']}** `{row['compound']:.3f}` _{row['source']}_")

with th2:
    st.markdown("**Most Negative 📉**")
    top_neg = (
        articles_df[articles_df["sentiment"] == "negative"]
        .nsmallest(5, "compound")[["title", "compound", "source"]]
    )
    for _, row in top_neg.iterrows():
        st.markdown(f"- **{row['title']}** `{row['compound']:.3f}` _{row['source']}_")

# ─────────────────────────────────────────────────────────────────
# News table
# ─────────────────────────────────────────────────────────────────
st.subheader("📰 All Articles")
display_cols = [c for c in ["title", "source", "sentiment", "compound",
                             "textblob_pol", "published_at"]
                if c in articles_df.columns]
st.dataframe(
    articles_df[display_cols].head(200),
    use_container_width=True,
    height=360,
)

# ─────────────────────────────────────────────────────────────────
# Keyword cloud table
# ─────────────────────────────────────────────────────────────────
if not keywords.empty:
    st.subheader("🔤 Top Keywords")
    st.dataframe(keywords.head(20), use_container_width=True, height=300)

# ─────────────────────────────────────────────────────────────────
# Downloads
# ─────────────────────────────────────────────────────────────────
st.subheader("⬇️ Download Results")
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "📄 Articles CSV",
        articles_df.to_csv(index=False).encode(),
        "articles_scored.csv",
        mime="text/csv",
    )
with dl2:
    st.download_button(
        "📊 Merged Analysis CSV",
        merged_df.to_csv(index=False).encode(),
        "merged_analysis.csv",
        mime="text/csv",
    )
with dl3:
    st.download_button(
        "📈 Daily Sentiment CSV",
        daily_sentiment.to_csv().encode(),
        "daily_sentiment.csv",
        mime="text/csv",
    )
