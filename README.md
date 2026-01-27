# Indian Equity Intelligence

A **free**, **open-source**, **local-first** application for analyzing Indian equities deeply and rationally.

> *"Is this stock suitable for this investor, under these assumptions — and what could invalidate the thesis?"*

## Philosophy

This tool provides **decision support, not decisions**. It:

- Rewards patience
- Penalizes weak governance
- Resists narrative bias
- Stays explainable at every step

**No price predictions. No trading signals. Just rational analysis.**

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/indian-equity-intelligence.git
cd indian-equity-intelligence

# Build and run with Docker Compose
docker-compose up --build

# Or run setup first to download initial data (optional)
docker-compose --profile setup run setup
docker-compose up
```

Open http://localhost:8501 in your browser.

### Option 2: Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-repo/indian-equity-intelligence.git
cd indian-equity-intelligence

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup (downloads initial data)
python setup.py

# Start the application
streamlit run app.py
```

Open http://localhost:8501 in your browser.

> **Note**: Docker or virtual environment is strongly recommended. It isolates this project's dependencies, preventing any conflicts with other installed packages.

## Features

### Core Analysis
- **Individual Stock Analysis**: Comprehensive analysis of any NSE-listed company
- **User Profile Matching**: Signals customized to YOUR expected return, risk appetite, and tenure
- **Five Analysis Dimensions**:
  - Legacy & Governance (promoter holding, pledges, auditor stability)
  - Financial Trajectory (growth, ROCE, cash flows, margins)
  - Valuation Context (PE/PB vs history and sector, PEG)
  - Market Behaviour (drawdowns, recovery, volatility)
  - ML-Assisted Context (company archetype classification)

### Signals
Contextual signals based on your profile:
- **BUY**: Stock aligns well with your investment profile
- **HOLD**: Mixed characteristics, monitor if already holding
- **AVOID BUYING**: Does not match your profile
- **SELL / EXIT**: Significant deterioration detected

### Mandatory Explainability
Every analysis includes:
- Clear reasoning for the signal
- **"Why This is NOT a Buy For You"** panel (shown even for BUY signals)
- Risk factors and thesis invalidators

### Advanced Features
- **Red Flag Alert System**: Automatic detection of governance and financial red flags
- **Time Travel Mode**: Re-analyze with only historical data (no future leakage)
- **Scenario Simulator**: Stress-test your thesis under various scenarios

## Project Structure

```
indian-equity-intelligence/
├── app.py                    # Main Streamlit application
├── setup.py                  # First-run setup script
├── requirements.txt          # Python dependencies
├── config/
│   └── settings.yaml         # Configuration and thresholds
├── data/
│   ├── cache/                # Parquet cache files
│   └── db/                   # DuckDB database
├── src/
│   ├── data/                 # Data layer
│   │   ├── sources.py        # Multi-source data ingestion
│   │   ├── database.py       # DuckDB manager
│   │   └── cache.py          # Cache manager
│   ├── engine/               # Analysis engines
│   │   ├── governance.py     # Governance analysis
│   │   ├── financial.py      # Financial analysis
│   │   ├── valuation.py      # Valuation analysis
│   │   ├── market.py         # Market behaviour analysis
│   │   └── ml_context.py     # ML-assisted context
│   ├── analysis/             # Signal and explainability
│   │   ├── signal_generator.py
│   │   ├── explainability.py
│   │   └── red_flags.py
│   ├── features/             # Advanced features
│   │   ├── time_travel.py
│   │   └── scenario_simulator.py
│   └── ui/                   # Streamlit components
│       ├── components.py
│       └── charts.py
└── tests/                    # Unit tests
```

## Data Sources

- Yahoo Finance (via yfinance)
- NSE Tools (nsetools)

Note: `jugaad-data` was removed due to dependency conflicts with yfinance. If needed, install separately with `pip install jugaad-data --no-deps`.

## Local-First Guarantee

- **100% offline operation** after initial setup
- **No telemetry** or tracking
- **No external API calls** during analysis
- All data stored locally

## Configuration

Edit `config/settings.yaml` to customize:
- Analysis thresholds
- Risk profile parameters
- Signal generation weights
- Red flag detection rules

## Limitations

- Data depends on free sources (may have delays)
- Shareholding pattern requires additional sources
- ML model requires data for training

## Non-Goals

This tool does NOT:
- Provide intraday or short-term signals
- Predict prices
- Execute trades
- Claim to be investment advice

## License

MIT License - see LICENSE file.

## Disclaimer

**This tool is for educational and informational purposes only.**

- It does NOT constitute investment advice
- Past performance does not guarantee future results
- Always do your own research
- Consult a qualified financial advisor

---

Built with ❤️ for the Indian retail investor community.
