"""
visualizer.py
-------------
Interactive Plotly visualizations for Market Sentiment Analyzer.

Charts
------
1. Sentiment vs. S&P 500 overlay (dual-axis)
2. Lagged cross-correlation bar chart (with CI whiskers)
3. Rolling correlation over time
4. Sentiment distribution histogram + pie
5. Regression scatter with OLS fit line
6. Correlation heatmap
7. Source distribution
8. Monthly sentiment trend
9. Candlestick + sentiment overlay

All charts saved as standalone HTML (interactive) and optional PNG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import OUTPUT_DIR, PALETTE
from logger import get_logger

log = get_logger(__name__)


class Visualizer:
    """
    Generates all charts for the Market Sentiment Analyzer.

    Parameters
    ----------
    output_dir : str | Path
        Where to save HTML/PNG files.
    """

    def __init__(self, output_dir: str | Path = OUTPUT_DIR):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # 1. Sentiment vs. SPX
    # ─────────────────────────────────────────────────────────────

    def sentiment_vs_spx(
        self, merged_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Dual-axis chart: sentiment 5d MA (left) vs. SPX close (right)."""
        df = merged_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Sentiment MA
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sentiment_ma5"],
                name="Sentiment (5d MA)",
                line=dict(color=PALETTE["sentiment"], width=2.5),
                fill="tozeroy", fillcolor=f"rgba(27,175,122,0.10)",
            ),
            secondary_y=False,
        )
        # Zero line
        fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1))

        # SPX close
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["close"],
                name="S&P 500 Close",
                line=dict(color=PALETTE["spx"], width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Sentiment Score (VADER)", secondary_y=False,
                         title_font_color=PALETTE["sentiment"])
        fig.update_yaxes(title_text="S&P 500 Close (USD)", secondary_y=True,
                         title_font_color=PALETTE["spx"])
        fig.update_layout(
            title="Market Sentiment vs. S&P 500 — 30-Day Window",
            template="plotly_dark",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            height=450,
        )

        return self._save(fig, "01_sentiment_vs_spx.html", save)

    # ─────────────────────────────────────────────────────────────
    # 2. Lagged correlation
    # ─────────────────────────────────────────────────────────────

    def lagged_correlation(
        self, lagged_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Bar chart of Pearson r at each lag with error bars (95% CI)."""
        if lagged_df.empty:
            return go.Figure()

        df = lagged_df.copy()
        colors = [
            PALETTE["positive"] if r >= 0 else PALETTE["negative"]
            for r in df["r"]
        ]

        # Error bar size
        err_y = None
        if "ci_upper" in df.columns and "ci_lower" in df.columns:
            ci_upper = df["ci_upper"].fillna(df["r"])
            ci_lower = df["ci_lower"].fillna(df["r"])
            err_y = dict(
                type="data",
                array=(ci_upper - df["r"]).tolist(),
                arrayminus=(df["r"] - ci_lower).tolist(),
                visible=True,
                color="#aaa",
            )

        fig = go.Figure(go.Bar(
            x=[f"Lag {l}d" for l in df["lag"]],
            y=df["r"],
            marker_color=colors,
            error_y=err_y,
            text=[
                f"r={r:.3f}<br>p={p:.3f}{'✱' if sig else ''}"
                for r, p, sig in zip(df["r"], df["p_value"], df["significant"])
            ],
            textposition="outside",
        ))

        fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1))
        fig.update_layout(
            title="Lagged Cross-Correlation: Sentiment → S&P 500 Return<br>"
                  "<sup>✱ = statistically significant (p < 0.05)</sup>",
            yaxis_title="Pearson r",
            yaxis_range=[-0.9, 0.9],
            template="plotly_dark",
            height=400,
        )

        return self._save(fig, "02_lagged_correlation.html", save)

    # ─────────────────────────────────────────────────────────────
    # 3. Rolling correlation
    # ─────────────────────────────────────────────────────────────

    def rolling_correlation(
        self, rolling_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Rolling 7-day Pearson r over time."""
        if rolling_df.empty:
            return go.Figure()

        df = rolling_df.copy()
        x  = pd.to_datetime(df["date"]) if "date" in df.columns else df.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=df["rolling_r"],
            name="Rolling r (7d)",
            line=dict(color=PALETTE["ma"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(74,58,167,0.10)",
        ))
        fig.add_hline(y=0,    line=dict(color="#888", dash="dash", width=1))
        fig.add_hline(y=0.3,  line=dict(color=PALETTE["positive"], dash="dot", width=0.8))
        fig.add_hline(y=-0.3, line=dict(color=PALETTE["negative"], dash="dot", width=0.8))

        fig.update_layout(
            title="Rolling Correlation: Sentiment vs. S&P 500 Daily Return",
            yaxis_title="Pearson r (7-day window)",
            yaxis_range=[-1.1, 1.1],
            template="plotly_dark",
            hovermode="x unified",
            height=400,
        )

        return self._save(fig, "03_rolling_correlation.html", save)

    # ─────────────────────────────────────────────────────────────
    # 4. Sentiment distribution
    # ─────────────────────────────────────────────────────────────

    def sentiment_distribution(
        self, articles_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Histogram + pie breakdown of VADER compound scores."""
        if articles_df.empty or "compound" not in articles_df.columns:
            return go.Figure()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Score Distribution", "Sentiment Breakdown"),
            specs=[[{"type": "histogram"}, {"type": "pie"}]],
        )

        color_map = {
            "positive": PALETTE["positive"],
            "negative": PALETTE["negative"],
            "neutral":  PALETTE["neutral"],
        }

        for label, color in color_map.items():
            subset = articles_df[articles_df["sentiment"] == label]["compound"]
            fig.add_trace(
                go.Histogram(
                    x=subset, name=label.capitalize(),
                    marker_color=color, opacity=0.8, nbinsx=20,
                ),
                row=1, col=1,
            )

        counts = articles_df["sentiment"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker_colors=[color_map.get(l, "#888") for l in counts.index],
                textinfo="percent+label",
                hole=0.35,
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title="VADER Sentiment Analysis — Article Distribution",
            template="plotly_dark",
            barmode="overlay",
            height=420,
        )

        return self._save(fig, "04_sentiment_distribution.html", save)

    # ─────────────────────────────────────────────────────────────
    # 5. Regression scatter
    # ─────────────────────────────────────────────────────────────

    def regression_scatter(
        self,
        merged_df: pd.DataFrame,
        reg_results: dict,
        save: bool = True,
    ) -> go.Figure:
        """Scatter: sentiment 5d MA vs. daily return with OLS line."""
        df = merged_df.dropna(subset=["sentiment_ma5", "daily_return"]).copy()
        if df.empty:
            return go.Figure()

        color_map = {
            "positive": PALETTE["positive"],
            "negative": PALETTE["negative"],
            "neutral":  PALETTE["neutral"],
        }
        dot_colors = (
            df["sentiment"].map(color_map).fillna("#aaa").tolist()
            if "sentiment" in df.columns else PALETTE["spx"]
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["sentiment_ma5"],
            y=df["daily_return"] * 100,
            mode="markers",
            marker=dict(color=dot_colors, size=8, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            name="Trading days",
            text=(df["date"].astype(str) if "date" in df.columns else None),
            hovertemplate="Sentiment: %{x:.3f}<br>Return: %{y:.2f}%<br>%{text}",
        ))

        # OLS line
        slope, intercept = reg_results.get("slope", 0), reg_results.get("intercept", 0)
        x_min, x_max = df["sentiment_ma5"].min(), df["sentiment_ma5"].max()
        xs = np.linspace(x_min, x_max, 100)
        ys = (slope * xs + intercept) * 100
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color="white", width=2),
            name=f"OLS  R²={reg_results.get('r_squared', 'N/A')}",
        ))

        fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
        fig.add_vline(x=0, line=dict(color="#555", dash="dash", width=0.8))

        fig.update_layout(
            title=f"Sentiment Score vs. S&P 500 Daily Return<br>"
                  f"<sup>OLS — R²={reg_results.get('r_squared')}  "
                  f"p={reg_results.get('p_value')}</sup>",
            xaxis_title="Sentiment Score (5d MA)",
            yaxis_title="S&P 500 Daily Return (%)",
            template="plotly_dark",
            height=480,
        )

        return self._save(fig, "05_regression_scatter.html", save)

    # ─────────────────────────────────────────────────────────────
    # 6. Correlation heatmap
    # ─────────────────────────────────────────────────────────────

    def correlation_heatmap(
        self, merged_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Heatmap of pairwise correlations for key numeric columns."""
        numeric_cols = [
            c for c in [
                "compound_mean", "sentiment_ma5", "daily_return",
                "volatility_5d", "article_count", "pct_positive", "pct_negative",
            ]
            if c in merged_df.columns
        ]
        if len(numeric_cols) < 2:
            return go.Figure()

        corr = merged_df[numeric_cols].corr().round(3)
        fig  = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title="Pairwise Correlation Heatmap",
            template="plotly_dark",
            height=500,
        )

        return self._save(fig, "06_heatmap.html", save)

    # ─────────────────────────────────────────────────────────────
    # 7. Source distribution
    # ─────────────────────────────────────────────────────────────

    def source_distribution(
        self, articles_df: pd.DataFrame, top_n: int = 12, save: bool = True
    ) -> go.Figure:
        """Bar chart of article count by news source."""
        if articles_df.empty or "source" not in articles_df.columns:
            return go.Figure()

        counts = (
            articles_df["source"].value_counts().head(top_n).reset_index()
        )
        counts.columns = ["source", "count"]

        fig = px.bar(
            counts, x="count", y="source", orientation="h",
            color="count", color_continuous_scale="Teal",
            title=f"Top {top_n} News Sources by Article Count",
        )
        fig.update_layout(template="plotly_dark", height=420, showlegend=False)
        fig.update_coloraxes(showscale=False)

        return self._save(fig, "07_source_distribution.html", save)

    # ─────────────────────────────────────────────────────────────
    # 8. Monthly sentiment trend
    # ─────────────────────────────────────────────────────────────

    def monthly_sentiment_trend(
        self, articles_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """Monthly average sentiment bar chart."""
        if articles_df.empty or "compound" not in articles_df.columns:
            return go.Figure()

        df = articles_df.copy()
        df["month"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.to_period("M")
        monthly = df.groupby("month")["compound"].mean().reset_index()
        monthly["month_str"] = monthly["month"].astype(str)
        monthly["color"] = monthly["compound"].apply(
            lambda s: PALETTE["positive"] if s >= 0 else PALETTE["negative"]
        )

        fig = go.Figure(go.Bar(
            x=monthly["month_str"], y=monthly["compound"],
            marker_color=monthly["color"],
            text=monthly["compound"].round(3),
            textposition="outside",
        ))
        fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1))
        fig.update_layout(
            title="Monthly Average Sentiment Score",
            yaxis_title="Avg VADER Compound",
            template="plotly_dark",
            height=380,
        )

        return self._save(fig, "08_monthly_sentiment.html", save)

    # ─────────────────────────────────────────────────────────────
    # 9. Candlestick + sentiment overlay
    # ─────────────────────────────────────────────────────────────

    def candlestick_with_sentiment(
        self, merged_df: pd.DataFrame, save: bool = True
    ) -> go.Figure:
        """OHLC candlestick chart with sentiment score overlaid."""
        df = merged_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        needed = {"open", "high", "low", "close"}.intersection(df.columns)
        if len(needed) < 4:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=("S&P 500 Price (Candlestick)", "Sentiment Score (5d MA)"),
        )

        close_only = (
            df[["open", "high", "low"]]
            .eq(df["close"], axis=0)
            .all()
            .all()
        )
        if close_only:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["close"],
                    mode="lines+markers",
                    name="S&P 500 Close",
                    line=dict(color=PALETTE["spx"], width=2.5),
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Candlestick(
                    x=df["date"],
                    open=df["open"], high=df["high"],
                    low=df["low"],   close=df["close"],
                    name="S&P 500",
                    increasing_line_color=PALETTE["positive"],
                    decreasing_line_color=PALETTE["negative"],
                ),
                row=1, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sentiment_ma5"],
                name="Sentiment 5d MA",
                line=dict(color=PALETTE["sentiment"], width=2),
                fill="tozeroy", fillcolor="rgba(27,175,122,0.12)",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, row=2, col=1, line=dict(color="#888", dash="dash", width=0.8))

        fig.update_layout(
            title=(
                "S&P 500 Close with Sentiment Overlay"
                if close_only else
                "S&P 500 Candlestick with Sentiment Overlay"
            ),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=620,
        )

        return self._save(fig, "09_candlestick_sentiment.html", save)

    # ─────────────────────────────────────────────────────────────
    # Full dashboard (all charts in one HTML)
    # ─────────────────────────────────────────────────────────────

    def dashboard(
        self,
        merged_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        results: dict,
        save: bool = True,
    ) -> go.Figure:
        """
        Combined dashboard: KPI cards + 6 charts in one HTML file.
        """
        from plotly.subplots import make_subplots

        pearson   = results.get("pearson", {})
        reg       = results.get("regression", {})
        best_lag  = results.get("best_lag", {})
        lagged    = results.get("lagged", pd.DataFrame())
        rolling   = results.get("rolling", pd.DataFrame())

        # Compose figure grid: 4 rows × 2 cols
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.09,
            horizontal_spacing=0.08,
            subplot_titles=(
                "Sentiment vs. S&P 500",  "",
                "Lagged Cross-Correlation", "Rolling Correlation",
                "Regression (OLS)",         "Sentiment Distribution",
                "Candlestick + Sentiment",  "",
            ),
            specs=[
                [{"secondary_y": True, "colspan": 2}, None],
                [{"type": "bar"},                      {"type": "scatter"}],
                [{"type": "scatter"},                  {"type": "pie"}],
                [{"colspan": 2},                       None],
            ],
        )

        df = merged_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ── Row 1: Sentiment vs. SPX ─────────────────────────────
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["sentiment_ma5"],
                       name="Sentiment 5d MA", line=dict(color=PALETTE["sentiment"], width=2),
                       fill="tozeroy", fillcolor="rgba(27,175,122,0.08)"),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["close"],
                       name="S&P 500", line=dict(color=PALETTE["spx"], width=2, dash="dot")),
            row=1, col=1, secondary_y=True,
        )

        # ── Row 2 left: Lagged correlation ────────────────────────
        if not lagged.empty:
            lag_colors = [PALETTE["positive"] if r >= 0 else PALETTE["negative"]
                          for r in lagged["r"]]
            fig.add_trace(
                go.Bar(x=[f"Lag {l}d" for l in lagged["lag"]], y=lagged["r"],
                       marker_color=lag_colors, name="Lag r"),
                row=2, col=1,
            )

        # ── Row 2 right: Rolling correlation ─────────────────────
        if not rolling.empty:
            rx = pd.to_datetime(rolling["date"]) if "date" in rolling.columns else rolling.index
            fig.add_trace(
                go.Scatter(x=rx, y=rolling["rolling_r"],
                           name="Rolling r (7d)", line=dict(color=PALETTE["ma"], width=2)),
                row=2, col=2,
            )

        # ── Row 3 left: Regression scatter ───────────────────────
        sc_df = merged_df.dropna(subset=["sentiment_ma5", "daily_return"])
        if not sc_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sc_df["sentiment_ma5"], y=sc_df["daily_return"] * 100,
                    mode="markers", marker=dict(color=PALETTE["spx"], size=7, opacity=0.7),
                    name="Days",
                ),
                row=3, col=1,
            )
            slope, icpt = reg.get("slope", 0), reg.get("intercept", 0)
            xs = np.linspace(sc_df["sentiment_ma5"].min(), sc_df["sentiment_ma5"].max(), 80)
            fig.add_trace(
                go.Scatter(x=xs, y=(slope * xs + icpt) * 100,
                           mode="lines", line=dict(color="white", width=1.8),
                           name=f"OLS R²={reg.get('r_squared', 'N/A')}"),
                row=3, col=1,
            )

        # ── Row 3 right: Pie chart ───────────────────────────────
        if not articles_df.empty and "sentiment" in articles_df.columns:
            counts = articles_df["sentiment"].value_counts()
            color_map = {"positive": PALETTE["positive"],
                         "negative": PALETTE["negative"],
                         "neutral":  PALETTE["neutral"]}
            fig.add_trace(
                go.Pie(labels=counts.index.tolist(), values=counts.values.tolist(),
                       marker_colors=[color_map.get(l, "#888") for l in counts.index],
                       textinfo="percent+label", hole=0.35, name="Sentiment"),
                row=3, col=2,
            )

        # ── Row 4: Candlestick ───────────────────────────────────
        if {"open", "high", "low", "close"}.issubset(df.columns):
            fig.add_trace(
                go.Candlestick(
                    x=df["date"], open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="OHLC",
                    increasing_line_color=PALETTE["positive"],
                    decreasing_line_color=PALETTE["negative"],
                ),
                row=4, col=1,
            )

        # ── Layout ────────────────────────────────────────────────
        r2      = reg.get("r_squared", "N/A")
        pear_r  = pearson.get("r", "N/A")
        lag_str = best_lag.get("lag_days", "N/A")

        fig.update_layout(
            title=dict(
                text=(
                    f"Market Sentiment Analyzer — Dashboard<br>"
                    f"<sup>Pearson r = {pear_r}  |  Best lag = {lag_str}d  |  "
                    f"R² = {r2}  |  "
                    f"{'✓ Significant' if pearson.get('significant') else '✗ Not significant'}</sup>"
                ),
                font_size=17,
            ),
            template="plotly_dark",
            height=1400,
            showlegend=True,
            xaxis4_rangeslider_visible=False,
            hovermode="x unified",
        )

        return self._save(fig, "00_dashboard.html", save)

    # ─────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────

    def _save(self, fig: go.Figure, filename: str, save: bool) -> go.Figure:
        if save:
            path = self.out / filename
            fig.write_html(str(path), include_plotlyjs="cdn")
            log.info("Saved chart → %s", path)
        return fig


# ─────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)
    n = 25
    dates = pd.date_range("2026-06-01", periods=n, tz="UTC")
    s     = np.random.uniform(-0.5, 0.7, n)
    ret   = s * 0.004 + np.random.normal(0, 0.006, n)
    close = 5000 + np.cumsum(ret * 500)

    merged = pd.DataFrame({
        "date": dates, "sentiment_ma5": s, "compound_mean": s,
        "daily_return": ret, "close": close,
        "open":  close * 0.998, "high": close * 1.004,
        "low":   close * 0.996, "volume": np.random.randint(1e9, 4e9, n),
    })
    articles = pd.DataFrame({
        "compound":  np.random.uniform(-1, 1, 80),
        "sentiment": np.random.choice(["positive", "negative", "neutral"], 80),
        "source":    np.random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"], 80),
        "published_at": pd.date_range("2026-06-01", periods=80, freq="9h", tz="UTC"),
    })

    fake_results = {
        "pearson":    {"r": 0.52, "p_value": 0.008, "significant": True,
                       "interpretation": "moderate positive",
                       "ci_lower": 0.18, "ci_upper": 0.76},
        "spearman":   {"r": 0.48, "p_value": 0.015, "significant": True},
        "kendall":    {"tau": 0.34, "p_value": 0.02, "significant": True},
        "lagged":     pd.DataFrame({
            "lag": [0,1,2,3,4,5], "r": [0.45,0.58,0.52,0.31,0.15,0.08],
            "p_value": [0.02]*6, "significant": [True]*4+[False]*2,
            "ci_lower": [0.1]*6, "ci_upper": [0.75]*6,
        }),
        "rolling":    merged[["date"]].assign(rolling_r=np.random.uniform(-0.5, 0.8, n)),
        "regression": {"slope": 0.004, "intercept": 0.001, "r_squared": 0.27,
                       "p_value": 0.008, "std_err": 0.001, "significant": True},
        "best_lag":   {"lag_days": 1, "r": 0.58, "p_value": 0.003, "significant": True},
        "volatility": {},
    }

    viz = Visualizer(output_dir="/tmp/viz_test")
    viz.dashboard(merged, articles, fake_results)
    viz.candlestick_with_sentiment(merged)
    viz.correlation_heatmap(merged)
    print("Charts saved to /tmp/viz_test/")
