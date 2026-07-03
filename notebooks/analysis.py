"""
notebooks/analysis.py
----------------------
Standalone script version of the Jupyter notebook.
Run this for step-by-step exploratory analysis with rich output.

To convert to a proper .ipynb:
    pip install jupytext
    jupytext --to notebook analysis.py

Or just run directly:
    python notebooks/analysis.py
"""

# %%
# ── Cell 1: Imports and setup ────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display   # works in both notebook and terminal

from news_fetcher       import NewsFetcher
from sentiment_scorer   import SentimentScorer
from market_data        import MarketData
from correlation_engine import CorrelationEngine
from visualizer         import Visualizer

print("✓ All imports successful")

# %%
# ── Cell 2: Fetch headlines ──────────────────────────────────────────
fetcher = NewsFetcher()
articles = fetcher.fetch(query="technology stocks S&P 500 earnings", days=30)
print(f"Fetched {len(articles)} articles")
print(articles[["title", "source", "published_at"]].head(10).to_string())

# %%
# ── Cell 3: Score sentiment ──────────────────────────────────────────
scorer = SentimentScorer()
articles = scorer.score(articles)

# Show top positive and negative headlines
print("\n── Top 3 Most Positive ──────────────────")
top_pos = articles.nlargest(3, "compound")[["title", "compound", "source"]]
print(top_pos.to_string(index=False))

print("\n── Top 3 Most Negative ──────────────────")
top_neg = articles.nsmallest(3, "compound")[["title", "compound", "source"]]
print(top_neg.to_string(index=False))

print("\n── Summary Stats ────────────────────────")
print(articles["compound"].describe().round(4))

# %%
# ── Cell 4: Daily aggregation ────────────────────────────────────────
daily = scorer.aggregate_daily(articles)
print("\nDaily sentiment (last 10 days):")
print(daily.tail(10).to_string())

# %%
# ── Cell 5: Quick sentiment trend plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily["compound_mean"], label="Daily avg", alpha=0.5, color="#1baf7a")
ax.plot(daily.index, daily["sentiment_ma5"], label="5-day MA", linewidth=2, color="#0f6e56")
ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
ax.fill_between(daily.index, daily["compound_mean"], 0,
                where=daily["compound_mean"] > 0, alpha=0.08, color="#1baf7a")
ax.fill_between(daily.index, daily["compound_mean"], 0,
                where=daily["compound_mean"] < 0, alpha=0.08, color="#e34948")
ax.set_title("Daily Sentiment Score Trend — Technology Sector")
ax.set_ylabel("VADER compound score")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/notebook_sentiment_trend.png", dpi=120)
plt.show()
print("Chart saved.")

# %%
# ── Cell 6: Market data ──────────────────────────────────────────────
md = MarketData(ticker="^GSPC")
market = md.fetch(days=30)
print(market[["date", "close", "daily_return", "cumulative_return"]].to_string())
print(f"\nS&P 500 30d return: {market['cumulative_return'].iloc[-1]*100:.2f}%")

# %%
# ── Cell 7: Merge datasets ───────────────────────────────────────────
merged = md.align_with_sentiment(market, daily)
print(f"\nMerged dataset: {len(merged)} overlapping trading days")
print(merged[["date", "compound_mean", "sentiment_ma5", "close", "daily_return"]].head(10).to_string())

# %%
# ── Cell 8: Correlation analysis ─────────────────────────────────────
engine = CorrelationEngine(merged)
results = engine.run_all()
engine.print_summary(results)

# %%
# ── Cell 9: Lagged correlation table ─────────────────────────────────
print("\nLagged Cross-Correlation Table:")
print(results["lagged"].to_string(index=False))
print("\nBest predictive lag:")
b = results["best_lag"]
print(f"  Lag {b['lag_days']} days  |  r = {b['r']}  |  p = {b['p_value']}  |  "
      f"{'Significant ✓' if b['significant'] else 'Not significant ✗'}")

# %%
# ── Cell 10: Full dashboard ───────────────────────────────────────────
viz = Visualizer(output_dir="outputs")
fig = viz.dashboard(merged, articles, results)
plt.show()
print("\nDashboard saved to outputs/00_dashboard.png")

# %%
# ── Cell 11: Scikit-learn regression deep dive ───────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X = merged[["sentiment_ma5"]].dropna()
y = merged.loc[X.index, "daily_return"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
print(f"\nLinear Regression CV Results (5-fold):")
print(f"  R² scores: {np.round(cv_scores, 3)}")
print(f"  Mean R²  : {cv_scores.mean():.4f}")
print(f"  Std R²   : {cv_scores.std():.4f}")

# %%
# ── Cell 12: Export results ───────────────────────────────────────────
articles.to_csv("outputs/articles_scored.csv", index=False)
daily.to_csv("outputs/daily_sentiment.csv")
merged.to_csv("outputs/merged_analysis.csv", index=False)
print("✓ All data exported to outputs/")
print("  articles_scored.csv  — article-level VADER scores")
print("  daily_sentiment.csv  — daily aggregated sentiment")
print("  merged_analysis.csv  — merged with S&P 500 data")
