# Indian Equity Intelligence

A **free**, **open-source**, **local-first** application for analyzing Indian equities with ML-powered insights.

> *"Is this stock suitable for this investor, under these assumptions — and what could invalidate the thesis?"*

## Quick Start (4 Steps)

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/indian-equity-intelligence.git
cd indian-equity-intelligence
```

### 2. Download ML Models

Run the interactive model downloader to choose models based on your PC specs:

```bash
python download_models.py
```

**Choose a preset based on your hardware:**

| Preset | RAM Required | What You Get |
|--------|--------------|--------------|
| 🚀 **Full** | 16GB+ | Chronos-T5-Base + LightGBM + Qwen2.5-3B |
| 💻 **Standard** | 8-16GB | Chronos-T5-Small + LightGBM + Qwen2.5-3B |
| 🪶 **Lite** | 4-8GB | Chronos-T5-Tiny + LightGBM (no LLM) |

### 3. Install Requirements

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 — **ML models initialize automatically!**

---

## What This Tool Does

### ML-Enhanced Stock Analysis

| Layer | Model | What It Does |
|-------|-------|--------------|
| **Forecaster** | Chronos-T5 | Predicts price trends for next 30 days |
| **Classifier** | LightGBM | Generates BUY/HOLD/AVOID/SELL signals |
| **Explainer** | Qwen2.5-3B | Writes natural language analysis |

All models run **100% locally** on your machine. No cloud APIs, no data sent anywhere.

### Analysis Dimensions

- **Governance**: Promoter holding, pledge ratio, auditor stability
- **Financial**: Revenue growth, ROCE, cash flows, margins
- **Valuation**: PE/PB vs history, PEG ratio
- **Market Behaviour**: Drawdowns, volatility, recovery patterns
- **ML Insights**: Price forecasts, AI-generated explanations

### Signals

Based on YOUR investment profile:
- **BUY**: Strong match with your criteria
- **HOLD**: Mixed signals, monitor
- **AVOID**: Doesn't fit your profile
- **SELL**: Significant concerns detected

---

## Hardware Requirements

| Setup | RAM | GPU | Models |
|-------|-----|-----|--------|
| **Minimum** | 8GB | None | Chronos-Tiny + LightGBM |
| **Recommended** | 16GB | Optional | Chronos-Base + LightGBM + Qwen2.5-3B |
| **Ideal** | 16GB+ | 8GB VRAM | All models with GPU acceleration |

---

## Alternative Setup Methods

### Docker

```bash
docker-compose up --build
```

### Manual Model Download

If the automatic downloader doesn't work:

```bash
# Create directories
mkdir -p models/{forecaster,classifier,explainer}

# Forecaster (choose one)
huggingface-cli download amazon/chronos-t5-base --local-dir models/forecaster/chronos-t5-base

# Explainer (optional)
mkdir -p models/explainer/qwen2.5-3b
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/explainer/qwen2.5-3b/model.gguf
```

See [ML_MODELS.md](ML_MODELS.md) for detailed instructions.

---

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
ml_config:
  enabled: true
  auto_initialize: true  # Models load on startup
  
  forecaster:
    model: "chronos-t5-base"  # or chronos-t5-small, chronos-t5-tiny
  
  classifier:
    model: "lightgbm"  # or catboost
  
  explainer:
    model: "qwen2.5-3b"  # or qwen2.5-7b, null to disable
```

---

## Features

### Core Features
- **Individual Stock Analysis**: Analyze any NSE-listed company
- **User Profile Matching**: Signals personalized to YOUR goals
- **Mandatory "Why NOT" Panel**: Shown even for BUY signals
- **Red Flag Detection**: Automatic governance/financial warnings

### ML Features
- **Price Predictions**: 5-day and 30-day trend forecasts
- **Signal Probabilities**: Confidence levels for each signal
- **AI Explanations**: Natural language analysis summaries
- **Trend Detection**: Bullish/Bearish/Neutral classification

### Advanced Features
- **Time Travel Mode**: Re-analyze with historical data only
- **Scenario Simulator**: Stress-test your investment thesis
- **Stock Suggestions**: Quick picks by category

---

## Project Structure

```
indian-equity-intelligence/
├── app.py                    # Main application
├── download_models.py        # Interactive model downloader
├── requirements.txt          # Python dependencies
├── config/settings.yaml      # Configuration
├── models/                   # ML models (download via script)
│   ├── forecaster/
│   ├── classifier/
│   └── explainer/
├── data/                     # Stock data (auto-downloaded)
└── src/                      # Source code
```

---

## Troubleshooting

### "Models not loading"
```bash
# Re-run the model downloader
python download_models.py
```

### "llama-cpp-python build fails"
```bash
# Install with pre-built wheel
pip install llama-cpp-python --prefer-binary

# Or for Apple Silicon
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### "Out of memory"
Choose a lighter preset in `download_models.py` or disable the LLM explainer in settings.yaml:
```yaml
explainer:
  model: null  # Uses rule-based explanations instead
```

---

## Privacy & Security

- **100% Local Processing**: No data leaves your machine
- **No Telemetry**: No tracking or analytics
- **Offline Capable**: Works without internet after setup
- **Open Source**: Full code transparency

---

## Disclaimer

**This tool is for educational purposes only.**

- Not investment advice
- Past performance ≠ future results
- Always do your own research
- Consult a qualified financial advisor

---

## License

MIT License - see LICENSE file.

---

Built with ❤️ for the Indian retail investor community.
