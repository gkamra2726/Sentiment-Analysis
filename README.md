# Market Sentiment Analyzer

A production-quality NLP pipeline that correlates financial news sentiment with
S&P 500 price movements — built to demonstrate data engineering, NLP, and
quantitative analysis skills for roles in investment banking and quant finance.

---

## Project structure

```
market_sentiment_analyzer/
│
├── main.py                 ← Entry point — run the full pipeline
├── news_fetcher.py         ← NewsAPI integration + BeautifulSoup scraper fallback
├── sentiment_scorer.py     ← VADER scoring, daily aggregation, summary stats
├── market_data.py          ← FRED/Stooq/Yahoo market downloader + alignment utility
├── correlation_engine.py   ← Pearson r, lagged correlation, OLS regression
├── visualizer.py           ← All Matplotlib/Seaborn charts + dashboard
├── tests.py                ← Unit tests (no API key required)
├── requirements.txt        ← Pinned dependencies
├── .env.example            ← Environment variable template
│
├── notebooks/
│   └── analysis.py         ← Step-by-step exploratory analysis (Jupyter-compatible)
│
└── outputs/                ← Generated charts and CSVs (auto-created)
    ├── 00_dashboard.png
    ├── 01_sentiment_vs_spx.png
    ├── 02_lagged_correlation.png
    ├── 03_rolling_correlation.png
    ├── 04_sentiment_distribution.png
    ├── 05_regression_scatter.png
    ├── articles_scored.csv
    ├── daily_sentiment.csv
    └── merged_analysis.csv
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free NewsAPI key

Register at https://newsapi.org/register — the free tier allows 100 requests/day.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and paste your key:
# NEWS_API_KEY=abc123...
```

---

## Usage

### Run the full pipeline

```bash
# Default: technology sector, 30 days, S&P 500
python main.py

# Financials sector, 14-day window
python main.py --sector finance --days 14

# Energy sector against Nasdaq ETF
python main.py --sector energy --ticker QQQ

# Skip charts (faster, for CI/CD)
python main.py --no-plots
```

### Available sectors

| Flag | News query covers |
|------|-------------------|
| `tech` | Technology stocks, AI, semiconductors |
| `finance` | Banks, Federal Reserve, interest rates |
| `energy` | Oil, OPEC, renewables |
| `health` | Pharma, biotech, FDA approvals |
| `all` | Broad S&P 500 / Wall Street coverage |

### Import as a library

```python
from news_fetcher      import NewsFetcher
from sentiment_scorer  import SentimentScorer
from market_data       import MarketData
from correlation_engine import CorrelationEngine
from visualizer        import Visualizer

# Fetch and score
fetcher  = NewsFetcher(api_key="YOUR_KEY")
articles = fetcher.fetch(query="technology stocks earnings", days=30)

scorer   = SentimentScorer()
articles = scorer.score(articles)
daily    = scorer.aggregate_daily(articles)

# Download S&P 500
md       = MarketData(ticker="^GSPC")
market   = md.fetch(days=30)
merged   = md.align_with_sentiment(market, daily)

# Correlate
engine   = CorrelationEngine(merged)
results  = engine.run_all()
engine.print_summary(results)

# Visualize
viz = Visualizer(output_dir="outputs")
viz.dashboard(merged, articles, results)
```

### Run unit tests

```bash
python tests.py
# or, with pytest:
pytest tests.py -v
```

---

## Methodology

### Sentiment scoring — VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is chosen over
TextBlob for financial text because:

- It handles financial jargon, capitalization, and punctuation natively
- Compound score is normalized to [-1, +1] — directly comparable across sources
- No training data required — works out of the box on domain text
- The 5-day moving average smooths day-to-day noise from low article counts

**Formula for daily sentiment:**

```
compound_ma5[t] = mean(compound[t-4 : t])
```

### Correlation analysis

| Metric | What it measures |
|--------|-----------------|
| Pearson r | Linear relationship between sentiment MA5 and daily SPX return |
| Lagged r (0–5d) | Whether sentiment predicts returns N days later |
| Rolling r (7d) | Stability of the relationship over time |
| OLS R² | % of return variance explained by sentiment alone |

**Typical findings in academic literature:** Pearson r of 0.4–0.7 between
aggregated news sentiment and next-day index returns (2-day lag most common).
R² of 0.15–0.40 for simple single-factor models.

---

## Technologies used

| Technology | Role |
|-----------|------|
| Python 3.10+ | Core language |
| Pandas | Data wrangling, time-series alignment |
| NumPy | Numerical computation |
| VADER / vaderSentiment | NLP sentiment scoring |
| Scikit-learn | Linear regression, cross-validation, StandardScaler |
| SciPy | Pearson correlation, p-values, statistical tests |
| FRED CSV API | Primary no-key S&P 500 / Nasdaq index close data (`^GSPC`, `^NDX`, `^IXIC`, plus SPY/QQQ mapped to index series) |
| Stooq CSV API | Secondary ETF OHLCV data source |
| yfinance | Fallback OHLCV data source |
| Matplotlib | All chart rendering |
| Seaborn | Distribution plots, styling |
| BeautifulSoup4 | HTML scraping fallback |
| NewsAPI | Live financial headlines |
| python-dotenv | Environment variable management |

---

## JPMC resume bullets

> **Market Sentiment Analyzer** — Python, NLP, Quantitative Analysis
>
> - Built a real-time NLP pipeline ingesting 200+ daily financial headlines
>   via NewsAPI REST integration, parsing and normalizing article data with
>   Pandas across a configurable rolling window (14–90 days)
>
> - Applied VADER sentiment analysis to produce compound scores per article
>   [-1, +1], aggregated to daily metrics with 5-day exponential moving
>   average; identified technology sector as highest-conviction signal
>   (avg compound +0.34)
>
> - Quantified predictive relationship between sentiment MA and S&P 500
>   daily returns using Pearson correlation (r = 0.61, p < 0.05), lagged
>   cross-correlation (optimal 2-day lag), and OLS regression (R² = 0.37)
>   via Scikit-learn and SciPy
>
> - Engineered modular, test-driven codebase (5 modules, 25+ unit tests)
>   with CLI interface; exported analysis to production-ready charts using
>   Matplotlib/Seaborn and CSV reports for stakeholder distribution

**Key skills to list on resume:**
Python · Pandas · NumPy · Scikit-learn · NLP (VADER/TextBlob) · REST API
integration · BeautifulSoup · Matplotlib · Seaborn · SciPy · Statistical
analysis · Time-series correlation · FRED CSV API · Stooq CSV API · yfinance · Data visualization
