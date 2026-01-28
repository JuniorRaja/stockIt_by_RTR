# Stocron by RTR

*The Indian Equity Intelligence*

A **local‑first, explainable stock research engine** for Indian equities. It combines governance, fundamentals, valuation, market behavior, macro regimes, and ML into one signal—while always showing the downside first so decisions stay grounded.

This is not a tip‑sheet. It’s a decision‑support system built for serious retail and advanced hobbyists who want transparent reasoning, not opaque recommendations.

## Architecture (High‑Level)

```
        Investor Profile (Return, Risk, Tenure)
                        │
                        ▼
    Data Layer (30y price + fundamentals + macro + delisted)
                        │
                        ▼
 ─────────────────────────────────────────────────┐
|               Analysis Engines                  |
|     Governance · Financial · Valuation · Market │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        ▼
                ML Pipeline (Optional)
    Chronos Forecaster → LightGBM Classifier → SHAP Explanations
                        │
                        ▼
      Final Signal + Confidence + “Why NOT to Buy”
```

## Installation & Setup

```bash
# 1) Clone
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# 2) Install dependencies
pip install -r requirements.txt

# 3) Setup environment
python 1-setup.py

# 4) Download stock data (~2,200 NSE stocks)
python 2-download_all_stocks.py

# 5) Delisted stocks (recommended for survivorship bias)
python scripts/download_delisted_stocks.py --export
python scripts/download_delisted_stocks.py --download --years 30

# 6) (Optional) Enable real crude oil data
export FRED_API_KEY="your_fred_api_key"

# 7) Download ML models (optional; required for ML forecaster/explainer)
python 3-download_models.py

# 8) Train classifier (required for ML signals)
python 4-train_classifier.py

# 9) Run the app
streamlit run app.py
```

Open http://localhost:8501

## Features

### 1) Profile‑Aware Signals
- Expected return, risk appetite, and holding period shape the BUY/HOLD/AVOID/SELL signal.
- A **“Why NOT to Buy”** section is mandatory for every stock, including BUY—use it to calibrate conviction.
- Benefit: keeps the recommendation aligned to *your* constraints rather than a generic “best stock”.

### 2) Governance Engine
- Promoter holding, pledge ratios, dividend consistency, auditor stability.
- Highlights red flags (pledge spikes, auditor churn, governance signals).
- Benefit: prevents “good numbers, bad stewardship” traps.

### 3) Financial Engine
- Revenue/PAT CAGR, ROCE, cash‑flow quality, leverage.
- **Investable universe filter** (e.g., sustained positive FCF) so weak cash‑flows are screened out early.
- Benefit: removes structurally weak businesses before any ML or valuation excitement kicks in.

### 4) Valuation Engine
- PE/PB, EV/EBITDA, PEG with historical percentile context.
- “Cheap” vs “expensive” is contextual, not absolute.
- Benefit: avoids paying peak multiples even for good companies.

### 5) Market Behavior Engine
- Drawdowns, recovery time, volatility regime, beta, Sharpe.
- Relative performance vs Nifty (1Y/3Y/5Y).
- Benefit: highlights resilience and downside behavior, not just upside returns.

### 6) Regime‑Aware Technicals
- RSI thresholds adapt to Bull/Bear/Sideways regimes.
- Prevents “overbought/oversold” misreads in regime shifts.
- Benefit: fewer false entries when the broader market structure changes.

### 7) Macro Context
- Repo rate, USD‑INR, crude oil, CPI inflation as first‑class features.
- Useful for 5–30 year cycles where macro dominates price action.
- Benefit: reduces tunnel‑vision on charts alone.

### 8) ML Insights (Optional)
- **Chronos forecaster** models 5–30 day trend direction.
- **LightGBM classifier** uses walk‑forward validation to avoid look‑ahead bias.
- **Risk‑adjusted targets** (Sharpe/Sortino) instead of raw price moves.
- **SHAP explanations** show which features pushed the decision up or down.
- Benefit: ML augments, not overrides—plus you get interpretable drivers.

### 9) Time Travel Mode
- Re‑run analysis as‑of a historical date using only data available then.
- Benefit: validates whether the logic would have helped *in real time*, not just hindsight.

### 10) Scenario Simulator
- Stress‑tests a stock across multiple scenarios (rates up, recession, commodity shock, currency volatility).
- **Good**: resilient/robust outcome with limited drawdown and fast recovery.
- **Bad**: fragile outcome with large drawdown, slow recovery, or thesis breaks.
- Benefit: helps you size positions and avoid asymmetric downside before entering.

## Disclaimer
This tool is for educational and research purposes only. It is not investment advice. Past performance does not guarantee future results. Always do your own research and consult a qualified financial advisor before investing.

## License
MIT License — see [LICENSE](LICENSE).
