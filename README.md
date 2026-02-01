# Stocron by RTR | The Ultimate Market Intelligence Engine

> **30+ Years of History. Zero Noise. Pure Alpha.**


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
* **The Explainer (Qwen2.5):** A local LLM that generates human-readable explanations for AI decisions.

> **GPU Acceleration:** All models support NVIDIA CUDA, Apple Metal (MPS), and Intel OpenVINO for 10-50x faster inference.

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
    │ ├────────────────┤ │
    │ │ Qwen2.5 (LLM)  │ │
    │ └────────────────┘ │
    │   ▲                │
    │   │ GPU/NPU        │
    │   │ Acceleration   │
    │ ┌─┴──────────────┐ │
    │ │ CUDA/MPS/OpenVINO│
    │ └────────────────┘ │
    └────────────────────┘
```

---

## Installation & Usage (Docker🐳)

> We strongly recommend running Stocron via Docker to avoid dependency hell.

---

### 🚀 Option A: GPU Mode (NVIDIA - Recommended)

**10-50x faster ML inference.** Supports RTX 20/30/40/50 series including Blackwell (RTX 5070/5080/5090).

#### Prerequisites
1. NVIDIA GPU with latest drivers (`nvidia-smi` should work)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

#### Copy-Paste Commands (GPU)

```bash
# Step 1: Clone the repository
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# Step 2: Build and start GPU container (first build takes ~15-20 mins)
docker-compose --profile gpu up -d --build

# Step 3: Build local database from bundled data
docker-compose exec stocron-by-rtr-gpu python 2-download_all_stocks.py --build-db-only

# Step 4: Download live stock data (optional but recommended)
docker-compose exec stocron-by-rtr-gpu python 2-download_all_stocks.py

# Step 5: Download ML models (choose option 2 for balanced setup)
docker-compose exec stocron-by-rtr-gpu python 3-download_models.py

# Step 6: Train the classifier
docker-compose exec stocron-by-rtr-gpu python 4-train_classifier.py

# Step 7: Verify GPU is working
docker-compose exec stocron-by-rtr-gpu python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

**Open the app:** http://localhost:8501

---

### 💻 Option B: CPU Mode (No GPU Required)

Works on any machine. ML inference will be slower but fully functional.

#### Prerequisites
1. Docker Desktop installed and running

#### Copy-Paste Commands (CPU)

```bash
# Step 1: Clone the repository
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# Step 2: Build and start CPU container
docker-compose --profile cpu up -d --build

# Step 3: Build local database from bundled data
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py --build-db-only

# Step 4: Download live stock data (optional but recommended)
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py

# Step 5: Download ML models (choose option 3 for lite setup on CPU)
docker-compose exec stocron-by-rtr python 3-download_models.py

# Step 6: Train the classifier
docker-compose exec stocron-by-rtr python 4-train_classifier.py
```

**Open the app:** http://localhost:8501

---

### 🔄 Switching Between GPU and CPU Modes

```bash
# Stop current container first
docker-compose --profile gpu down    # if running GPU mode
docker-compose --profile cpu down    # if running CPU mode

# Then start the other mode
docker-compose --profile gpu up -d   # switch to GPU
docker-compose --profile cpu up -d   # switch to CPU
```

---

### 📥 Optional: Download Delisted Stocks (for Survivorship Bias Analysis)

```bash
# GPU mode
docker-compose exec stocron-by-rtr-gpu python scripts/download_delisted_stocks.py --export
docker-compose exec stocron-by-rtr-gpu python scripts/download_delisted_stocks.py --download --years 30
docker-compose exec stocron-by-rtr-gpu python 2-download_all_stocks.py --build-db-only

# CPU mode (replace container name)
docker-compose exec stocron-by-rtr python scripts/download_delisted_stocks.py --export
docker-compose exec stocron-by-rtr python scripts/download_delisted_stocks.py --download --years 30
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py --build-db-only
```

---

### 🍎 Apple Silicon (M1/M2/M3/M4)

Apple Silicon uses Metal Performance Shaders (MPS). Run natively (not in Docker) for best GPU performance:

```bash
# Install dependencies
pip install -r requirements.txt

# Enable Metal for Qwen LLM
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# Run the app
streamlit run app.py
```

---

### ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA not available` | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker |
| `nvidia-smi` not found | Install NVIDIA drivers from [nvidia.com](https://nvidia.com) |
| `Out of memory` | Use smaller model: choose option [3] LITE in `3-download_models.py` |
| Port 8501 already in use | Stop other containers: `docker-compose --profile gpu down` or `docker-compose --profile cpu down` |
| Feature mismatch error | Retrain classifier: `python 4-train_classifier.py` |
| RTX 50-series not detected | The GPU image uses PyTorch nightly with CUDA 12.8 for Blackwell support. Rebuild: `docker-compose --profile gpu up -d --build` |

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
