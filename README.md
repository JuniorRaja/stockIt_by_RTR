# 🚀 Stocron by RTR | The Ultimate Market Intelligence Engine

> **30 Years of History. Zero Noise. Pure Alpha.**


[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/AI-Chronos--T5%20%2B%20LightGBM-orange?style=for-the-badge)](https://huggingface.co/amazon/chronos-t5-tiny)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📉 The Problem
Retail investors are playing a rigged game. Most analysis tools suffer from three critical flaws:
1.  **Recency Bias:** They only look at the last 5-10 years of a Bull market.
2.  **Survivorship Bias:** They ignore delisted companies, making history look safer than it actually was.
3.  **Linear Thinking:** They use simple indicators (RSI, MA) in a complex, non-linear macro environment.

You cannot build generational wealth by just "looking at the chart." You need to understand the **Regime**, the **Fundamentals**, and the **Macro-Economic** backdrop.

## 🛡️ Why This Tool Exists
**Stocron** was built for the **RTR Unfiltered** ecosystem to answer one question:
*"If I had the same data, tools, and computing power as a hedge fund, but with 30 years of unfiltered Indian market context, how would I trade?"*

This is not just a screener. It is a **Time Machine**. It allows you to validate strategies across decades, factoring in inflation, oil prices, and corporate governance failures.

---

## ⚡ Super Powers

### 1. The "Dual-Brain" AI Core 🧠
We don't rely on a single model. Stocron uses a hybrid architecture:
* **The Forecaster (Chronos-T5):** A Transformer-based model (pretrained by Amazon) that treats stock charts like a language to predict future price sequences.
* **The Classifier (LightGBM):** A gradient-boosting decision engine that analyzes hundreds of features (Financials, Macro, Technicals) to generate a binary `BUY`/`HOLD` signal.

### 2. Time Travel & Survivorship ⏳
Most backtests are fake because they test on companies that exist *today*.
* **Delisted Database:** We track companies that failed, ensuring your strategy survives the worst.
* **Time Travel Engine:** Go back to Jan 1st, 2008. The system "forgets" the future, forcing the AI to trade only on what it knew then.

### 3. Scenario Simulator 🌪️
Don't just predict; prepare.
* *What if Crude Oil hits $120?*
* *What if the Repo Rate jumps to 8%?*
The simulator stresses your portfolio against hypothetical macro-economic shocks.

### 4. Governance Guard 🕵️
The tool doesn't just chase profits; it filters out fraud.
* **Beneish M-Score:** Detects earnings manipulation.
* **Altman Z-Score:** Predicts bankruptcy risk.
* **Piotroski F-Score:** Measures fundamental strength.

---

## Architecture 🏗️

The system is containerized for stability and reproducibility.

```text
       ┌──────────────┐
       │     USER     │
       └──────┬───────┘
              │ (Browser)
              ▼
    ┌────────────────────┐
    │    STREAMLIT UI    │
    └─────────┬──────────┘
              │ (Request)
              ▼
    ┌────────────────────┐          ┌──────────────────┐
    │  ANALYSIS ENGINE   │◄─────────│   DATA LAKE DB   │
    └─────────┬──────────┘          │  (JSON / CSV)    │
              │                     └─────────▲────────┘
              │ (Inference)                   │
              │                               │ (Fetch)
    ┌─────────▼──────────┐          ┌─────────┴────────┐
    │     ML MODELS      │          │   INTERNET / NSE │
    │ ┌────────────────┐ │          └──────────────────┘
    │ │ Chronos-T5     │ │
    │ ├────────────────┤ │
    │ │ LightGBM       │ │
    │ └────────────────┘ │
    └────────────────────┘
```

---

## Installation & Usage (Docker🐳)
> We strongly recommend running Stocron via Docker to avoid dependency hell.

### Prerequisites
Docker Desktop installed and running.

```bash
# 1) Clone
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# 2) Install dependencies
pip install -r requirements.txt

# 3) Setup environment
python 1-setup.py

# 4) Hydrate the Data Lake - First Time Only (~2,200 NSE stocks)
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

# 9) Run the App through Container
docker-compose up --build    

Open👉 http://localhost:8501
```
---

## ML Logic & Explainability 🧩
We believe in "Unfiltered" truth. The AI shouldn't be a black box.

SHAP Integration: We use SHAP (SHapley Additive exPlanations) to break down every signal.

Example Output: "The model is Bullish because 'ROE > 15%' (+20 impact) and 'Oil Prices Dropped' (+10 impact), despite 'RSI being Overbought' (-5 impact)."

## Disclaimer 📜
This tool is for **educational and research purposes only**. It is built for the RTR Unfiltered community to analyze market logic. It is NOT financial advice. Markets are subject to risk. *Use your own brain before making financial decisions*.

## License
MIT License — see [LICENSE](LICENSE).


_**Built with 🧠 by RTR.**_