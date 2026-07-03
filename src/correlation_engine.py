"""
correlation_engine.py
----------------------
Full statistical analysis between news sentiment and market returns.

Analyses included
-----------------
• Pearson, Spearman, Kendall correlations (with confidence intervals)
• Lagged cross-correlation (0 … MAX_LAG_DAYS)
• Rolling correlation over time
• OLS linear regression (slope, intercept, R², p-value, SE)
• Volatility analysis (sentiment std vs. return std)
• Sentiment momentum analysis

Public API
----------
    engine  = CorrelationEngine(merged_df)
    results = engine.run_all()
    engine.print_summary(results)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import MAX_LAG_DAYS, MIN_OBSERVATIONS, ROLLING_WINDOW, SIGNIFICANCE_LVL
from logger import get_logger

log = get_logger(__name__)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson r and p-value; returns (nan, nan) if insufficient data."""
    if len(x) < MIN_OBSERVATIONS:
        return float("nan"), float("nan")
    try:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def _confidence_interval(r: float, n: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Fisher Z-transform confidence interval for Pearson r.
    Returns (lower, upper).
    """
    if n < 4 or np.isnan(r):
        return float("nan"), float("nan")
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf((1 + alpha) / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


class CorrelationEngine:
    """
    Statistical analysis between sentiment scores and market returns.

    Parameters
    ----------
    df            : merged DataFrame from MarketData.align_with_sentiment()
    sentiment_col : column to use as the sentiment signal
    return_col    : column to use as the market signal
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment_ma5",
        return_col:    str = "daily_return",
    ):
        if df.empty:
            log.warning("CorrelationEngine received empty DataFrame")
        self.df            = df.dropna(subset=[sentiment_col, return_col]).copy()
        self.sentiment_col = sentiment_col
        self.return_col    = return_col
        self.x             = self.df[sentiment_col].values.astype(float)
        self.y             = self.df[return_col].values.astype(float)
        log.debug("CorrelationEngine: %d observations", len(self.x))

    # ─────────────────────────────────────────────────────────────
    # Runner
    # ─────────────────────────────────────────────────────────────

    def run_all(self) -> Dict:
        """Run all analyses. Returns dict of results."""
        if len(self.x) < MIN_OBSERVATIONS:
            log.warning(
                "Insufficient data (%d rows < %d). Returning empty results.",
                len(self.x), MIN_OBSERVATIONS,
            )
            return self._empty_results()

        results = {}
        results["pearson"]    = self.pearson()
        results["spearman"]   = self.spearman()
        results["kendall"]    = self.kendall()
        results["lagged"]     = self.lagged_correlation(max_lag=MAX_LAG_DAYS)
        results["rolling"]    = self.rolling_correlation(window=ROLLING_WINDOW)
        results["regression"] = self.linear_regression()
        results["volatility"] = self.volatility_analysis()
        results["best_lag"]   = self._best_lag(results["lagged"])
        return results

    # ─────────────────────────────────────────────────────────────
    # Correlations
    # ─────────────────────────────────────────────────────────────

    def pearson(self) -> dict:
        """Pearson r with 95% CI."""
        r, p = _safe_pearson(self.x, self.y)
        lo, hi = _confidence_interval(r, len(self.x))
        return {
            "r":              round(r, 4),
            "p_value":        round(p, 4),
            "significant":    bool(p < SIGNIFICANCE_LVL) if not np.isnan(p) else False,
            "ci_lower":       round(lo, 4) if not np.isnan(lo) else None,
            "ci_upper":       round(hi, 4) if not np.isnan(hi) else None,
            "interpretation": self._interpret_r(r),
        }

    def spearman(self) -> dict:
        """Spearman rank correlation."""
        if len(self.x) < MIN_OBSERVATIONS:
            return {"r": float("nan"), "p_value": float("nan"), "significant": False}
        try:
            r, p = stats.spearmanr(self.x, self.y)
        except Exception:
            r, p = float("nan"), float("nan")
        return {
            "r":           round(float(r), 4),
            "p_value":     round(float(p), 4),
            "significant": bool(float(p) < SIGNIFICANCE_LVL),
        }

    def kendall(self) -> dict:
        """Kendall tau correlation."""
        if len(self.x) < MIN_OBSERVATIONS:
            return {"tau": float("nan"), "p_value": float("nan"), "significant": False}
        try:
            tau, p = stats.kendalltau(self.x, self.y)
        except Exception:
            tau, p = float("nan"), float("nan")
        return {
            "tau":         round(float(tau), 4),
            "p_value":     round(float(p), 4),
            "significant": bool(float(p) < SIGNIFICANCE_LVL),
        }

    # ─────────────────────────────────────────────────────────────
    # Lagged cross-correlation
    # ─────────────────────────────────────────────────────────────

    def lagged_correlation(self, max_lag: int = 5) -> pd.DataFrame:
        """
        Pearson r at sentiment lags 0 … max_lag days.
        Positive lag = sentiment precedes price (predictive signal).

        Returns DataFrame: lag, r, p_value, significant, ci_lower, ci_upper
        """
        records = []
        for lag in range(0, max_lag + 1):
            x_lag = self.x[:-lag] if lag > 0 else self.x
            y_lag = self.y[lag:]  if lag > 0 else self.y

            if len(x_lag) < MIN_OBSERVATIONS:
                continue

            r, p = _safe_pearson(x_lag, y_lag)
            lo, hi = _confidence_interval(r, len(x_lag))
            records.append({
                "lag":         lag,
                "r":           round(r, 4),
                "p_value":     round(p, 4),
                "significant": bool(p < SIGNIFICANCE_LVL) if not np.isnan(p) else False,
                "ci_lower":    round(lo, 4) if not np.isnan(lo) else None,
                "ci_upper":    round(hi, 4) if not np.isnan(hi) else None,
            })

        return pd.DataFrame(records) if records else pd.DataFrame(
            columns=["lag", "r", "p_value", "significant", "ci_lower", "ci_upper"]
        )

    # ─────────────────────────────────────────────────────────────
    # Rolling correlation
    # ─────────────────────────────────────────────────────────────

    def rolling_correlation(self, window: int = 7) -> pd.DataFrame:
        """Rolling Pearson r in a sliding window."""
        df = self.df.copy()
        df["rolling_r"] = (
            df[self.sentiment_col]
            .rolling(window=window, min_periods=max(3, window // 2))
            .corr(df[self.return_col])
            .round(4)
        )
        cols = ["date", "rolling_r"] if "date" in df.columns else ["rolling_r"]
        return df[cols].dropna()

    # ─────────────────────────────────────────────────────────────
    # Linear regression
    # ─────────────────────────────────────────────────────────────

    def linear_regression(self) -> dict:
        """OLS: daily_return ~ sentiment_ma5."""
        if len(self.x) < MIN_OBSERVATIONS:
            return self._empty_regression()

        try:
            # scipy for p-value and SE; sklearn for R²
            slope, intercept, r, p, se = stats.linregress(self.x, self.y)
            scaler = StandardScaler()
            x_sc   = scaler.fit_transform(self.x.reshape(-1, 1))
            model  = LinearRegression().fit(x_sc, self.y)
            y_pred = model.predict(x_sc)
            r2     = r2_score(self.y, y_pred)

            return {
                "slope":       round(float(slope), 6),
                "intercept":   round(float(intercept), 6),
                "r_squared":   round(float(r2), 4),
                "p_value":     round(float(p), 4),
                "std_err":     round(float(se), 6),
                "significant": bool(float(p) < SIGNIFICANCE_LVL),
            }
        except Exception as exc:
            log.error("Regression failed: %s", exc)
            return self._empty_regression()

    # ─────────────────────────────────────────────────────────────
    # Volatility analysis
    # ─────────────────────────────────────────────────────────────

    def volatility_analysis(self) -> dict:
        """
        Compare sentiment volatility (std of compound_std) to market volatility.
        Also computes correlation between daily sentiment std and return abs value.
        """
        result = {}
        df = self.df

        if "compound_std" in df.columns:
            result["sentiment_volatility"] = round(float(df["compound_std"].mean()), 4)

        if "volatility_5d" in df.columns:
            result["market_volatility_5d"] = round(float(df["volatility_5d"].mean()), 4)

        if "compound_std" in df.columns and "daily_return" in df.columns:
            valid = df[["compound_std", "daily_return"]].dropna()
            if len(valid) >= MIN_OBSERVATIONS:
                r, p = _safe_pearson(
                    valid["compound_std"].values,
                    valid["daily_return"].abs().values,
                )
                result["vol_correlation_r"]  = round(r, 4)
                result["vol_correlation_p"]  = round(p, 4)

        return result

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _best_lag(self, lagged_df: pd.DataFrame) -> dict:
        if lagged_df.empty:
            return {}
        idx = lagged_df["r"].abs().idxmax()
        row = lagged_df.loc[idx]
        return {
            "lag_days":    int(row["lag"]),
            "r":           round(float(row["r"]), 4),
            "p_value":     round(float(row["p_value"]), 4),
            "significant": bool(row["significant"]),
        }

    @staticmethod
    def _interpret_r(r: float) -> str:
        if np.isnan(r):
            return "insufficient data"
        a = abs(r)
        if a >= 0.7:   strength = "strong"
        elif a >= 0.4: strength = "moderate"
        elif a >= 0.2: strength = "weak"
        else:           strength = "negligible"
        sign = "positive" if r >= 0 else "negative"
        return f"{strength} {sign} correlation"

    def _empty_results(self) -> Dict:
        empty_corr = {"r": float("nan"), "p_value": float("nan"),
                      "significant": False, "interpretation": "insufficient data"}
        return {
            "pearson":    empty_corr,
            "spearman":   {"r": float("nan"), "p_value": float("nan"), "significant": False},
            "kendall":    {"tau": float("nan"), "p_value": float("nan"), "significant": False},
            "lagged":     pd.DataFrame(columns=["lag", "r", "p_value", "significant"]),
            "rolling":    pd.DataFrame(columns=["rolling_r"]),
            "regression": self._empty_regression(),
            "volatility": {},
            "best_lag":   {},
        }

    @staticmethod
    def _empty_regression() -> dict:
        return {
            "slope": float("nan"), "intercept": float("nan"),
            "r_squared": float("nan"), "p_value": float("nan"),
            "std_err": float("nan"), "significant": False,
        }

    def print_summary(self, results: dict) -> None:
        """Pretty-print all correlation results."""
        p  = results.get("pearson", {})
        sp = results.get("spearman", {})
        k  = results.get("kendall", {})
        r  = results.get("regression", {})
        b  = results.get("best_lag", {})
        v  = results.get("volatility", {})

        lines = [
            "=" * 52,
            "  CORRELATION ANALYSIS SUMMARY",
            "=" * 52,
            f"  Observations   : {len(self.x)}",
            f"  Sentiment col  : {self.sentiment_col}",
            f"  Return col     : {self.return_col}",
            "",
            f"  Pearson r      : {p.get('r', 'N/A')}  ({p.get('interpretation', '')})",
            f"  Pearson p      : {p.get('p_value', 'N/A')}  "
            f"{'✓ significant' if p.get('significant') else '✗ not significant'}",
            f"  95% CI         : [{p.get('ci_lower', 'N/A')}, {p.get('ci_upper', 'N/A')}]",
            "",
            f"  Spearman r     : {sp.get('r', 'N/A')}  p={sp.get('p_value', 'N/A')}",
            f"  Kendall tau    : {k.get('tau', 'N/A')}  p={k.get('p_value', 'N/A')}",
            "",
            f"  R²             : {r.get('r_squared', 'N/A')}",
            f"  Slope          : {r.get('slope', 'N/A')}",
            f"  Intercept      : {r.get('intercept', 'N/A')}",
            "",
            f"  Best lag       : {b.get('lag_days', 'N/A')} day(s)  r={b.get('r', 'N/A')}",
            "=" * 52,
        ]
        print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 30
    x = np.random.uniform(-0.5, 0.8, n)
    y = x * 0.003 + np.random.normal(0, 0.005, n)

    df = pd.DataFrame({
        "date":          pd.date_range("2026-06-01", periods=n),
        "sentiment_ma5": x,
        "daily_return":  y,
        "compound_std":  np.abs(np.random.normal(0.1, 0.05, n)),
    })

    engine  = CorrelationEngine(df)
    results = engine.run_all()
    engine.print_summary(results)
    print("\nSpearman:", results["spearman"])
    print("Kendall:", results["kendall"])
    print("Lagged:\n", results["lagged"].to_string(index=False))
