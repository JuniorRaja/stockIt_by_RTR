# Indian Equity Intelligence

A local-first, ML-powered stock analysis tool for Indian equities. It combines governance, fundamentals, valuation, market behavior, and macro context into a single, explainable signal.

## What It Does
- Personalized stock signals (BUY/HOLD/AVOID/SELL) based on your risk profile
- Mandatory “Why NOT to Buy” risks for every stock
- Regime-aware RSI thresholds (Bull/Bear/Sideways)
- Macro factors (repo rate, USD-INR, crude oil, CPI)
- ML classifier trained with walk‑forward validation and risk‑adjusted targets
- SHAP‑based explanations when ML is enabled
- Time‑travel and scenario analysis (optional)

## Installation (Exact Order)

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/indian-equity-intelligence.git
cd indian-equity-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
python 1-setup.py

# 4. Download stock data (~2,200 NSE stocks)
python 2-download_all_stocks.py

# 5. Download delisted stocks (optional but recommended)
python scripts/download_delisted_stocks.py --export

# 6. (Optional) Enable real crude oil data
# Set your FRED API key to avoid synthetic data warnings
export FRED_API_KEY="your_fred_api_key"

# 7. Download ML models (optional; required for ML forecaster/explainer)
python 3-download_models.py

# 8. Train the classifier (required for ML signals)
python 4-train_classifier.py

# 9. Run the app
streamlit run app.py
```

Open http://localhost:8501

## Notes
- Steps 5–8 are optional if you only want rule‑based analysis.
- If you skip step 7, you can still run the app without ML.
- If you skip step 5, survivorship bias handling is limited to active stocks only.

## Disclaimer
This tool is for educational and research purposes only. It is not investment advice. Past performance does not guarantee future results. Always do your own research and consult a qualified financial advisor before investing.

## License
MIT License — see [LICENSE](LICENSE).
