"""
Indian Equity Intelligence - Main Application
A local-first, explainable, long-term stock analysis tool.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config, get_config
from src.data.sources import DataSourceManager
from src.data.database import DatabaseManager
from src.data.cache import CacheManager
from src.engine.governance import GovernanceAnalyzer
from src.engine.financial import FinancialAnalyzer
from src.engine.valuation import ValuationAnalyzer
from src.engine.market import MarketBehaviourAnalyzer
from src.engine.ml_context import MLContextAnalyzer
from src.analysis.signal_generator import SignalGenerator, UserProfile
from src.analysis.explainability import ExplainabilityEngine
from src.analysis.red_flags import RedFlagDetector
from src.features.time_travel import TimeTravelEngine
from src.features.scenario_simulator import ScenarioSimulator
from src.ui.components import render_user_profile, render_signal_badge, render_why_not_buy, render_red_flags, render_dimension_scores, render_footer
from src.ui.charts import create_price_chart, create_score_radar, create_drawdown_chart

# ML Models Integration
from src.ml_models.ensemble import MLEnsemble, EnsembleConfig, create_ensemble_from_config
from src.ml_models.base import ModelType

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Indian Equity Intelligence", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { padding: 0 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div[data-testid="metric-container"] { background-color: #262730; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


class IndianEquityIntelligence:
    def __init__(self):
        self.config = load_config()
        self._data_manager = None
        self._db_manager = None
        self.governance = GovernanceAnalyzer()
        self.financial = FinancialAnalyzer()
        self.valuation = ValuationAnalyzer()
        self.market = MarketBehaviourAnalyzer()
        self.ml_context = MLContextAnalyzer()
        self.signal_gen = SignalGenerator()
        self.explainer = ExplainabilityEngine()
        self.red_flag = RedFlagDetector()
        self.time_travel = TimeTravelEngine()
        self.scenario_sim = ScenarioSimulator()
        
        # Initialize ML Ensemble (3-layer architecture)
        self._ml_ensemble = None
        self._ml_initialized = False
        self._ml_enabled = self.config.get('ml_config', {}).get('enabled', True)
    
    @property
    def data_manager(self):
        if self._data_manager is None:
            self._data_manager = DataSourceManager()
        return self._data_manager
    
    @property
    def ml_ensemble(self):
        """Lazy initialization of ML ensemble."""
        if self._ml_ensemble is None and self._ml_enabled:
            try:
                ml_config = self.config.get('ml_config', {})
                
                # Build ensemble config from settings
                ensemble_config_dict = {
                    "forecaster": ml_config.get('forecaster', {}).get('model'),
                    "classifier": ml_config.get('classifier', {}).get('model'),
                    "explainer": ml_config.get('explainer', {}).get('model'),
                    "device": ml_config.get('device', 'auto'),
                    "models_dir": ml_config.get('models_dir', 'models'),
                }
                
                self._ml_ensemble = create_ensemble_from_config(ensemble_config_dict)
                
            except Exception as e:
                logger.warning(f"Failed to create ML ensemble: {e}")
                self._ml_enabled = False
        
        return self._ml_ensemble
    
    def initialize_ml(self):
        """Initialize ML models (call explicitly when ready to load into memory)."""
        if self._ml_initialized or not self._ml_enabled:
            return self._ml_initialized, []
        
        if self.ml_ensemble:
            try:
                success, messages = self.ml_ensemble.initialize(load_models=True)
                self._ml_initialized = success
                return success, messages
            except Exception as e:
                logger.error(f"ML initialization failed: {e}")
                return False, [str(e)]
        
        return False, ["ML ensemble not configured"]
    
    def get_ml_status(self) -> dict:
        """Get status of ML models."""
        if not self._ml_enabled:
            return {"enabled": False, "reason": "ML disabled in config"}
        
        if not self.ml_ensemble:
            return {"enabled": True, "initialized": False, "reason": "Not yet initialized"}
        
        return {
            "enabled": True,
            "initialized": self._ml_initialized,
            **self.ml_ensemble.get_status()
        }
    
    def fetch_data(self, symbol: str):
        with st.spinner(f"Fetching data for {symbol}..."):
            info = self.data_manager.get_stock_info(symbol)
            if not info:
                st.error(f"Could not find: {symbol}")
                return None
            prices = self.data_manager.get_price_history(symbol, years=10)
            if prices is None or prices.empty:
                st.error(f"No price history for {symbol}")
                return None
            financials = self.data_manager.get_financials(symbol) or {}
            nifty = self.data_manager.get_price_history("NIFTY", years=10)
            return {'stock_info': info, 'price_history': prices, 'financials': financials,
                    'nifty_history': nifty, 'shareholding': pd.DataFrame(), 'dividends': pd.DataFrame()}
    
    def run_analysis(self, symbol: str, profile: UserProfile, data: dict):
        with st.spinner("Running analysis..."):
            info = data['stock_info']
            prices = data['price_history']
            fins = data['financials']
            nifty = data['nifty_history']
            share = data.get('shareholding', pd.DataFrame())
            divs = data.get('dividends', pd.DataFrame())
            
            # Run traditional analysis engines
            gov = self.governance.analyze(symbol, info, share, divs)
            fin = self.financial.analyze(symbol, fins, info)
            val = self.valuation.analyze(symbol, info, prices, fins, earnings_growth=fin.pat_cagr_5y)
            mkt = self.market.analyze(symbol, prices, nifty)
            red_flags = self.red_flag.detect_all_flags(symbol, info, share, fins)
            
            # Generate initial signal from rule-based system
            signal = self.signal_gen.generate_signal(
                profile, gov, fin, val, mkt,
                red_flags=[{'severity': r.severity, 'description': r.description} for r in red_flags]
            )
            
            # Run ML-enhanced analysis if available
            ml_prediction = None
            if self._ml_enabled and self.ml_ensemble and self._ml_initialized:
                try:
                    with st.spinner("Running ML-enhanced analysis..."):
                        ml_prediction = self.ml_ensemble.predict(
                            symbol=symbol,
                            prices=prices,
                            governance_result=gov,
                            financial_result=fin,
                            valuation_result=val,
                            market_result=mkt,
                            company_name=info.get('name', symbol),
                            current_signal=signal.signal,
                            current_confidence=signal.confidence,
                            red_flags=[r.description for r in red_flags],
                        )
                        
                        # Blend ML score with rule-based score
                        if ml_prediction and ml_prediction.success:
                            ml_weight = self.config.get('ml_config', {}).get('ml_weight_in_composite', 0.15)
                            
                            # Update composite score with ML contribution
                            blended_score = (
                                signal.composite_score * (1 - ml_weight) + 
                                ml_prediction.ml_score * ml_weight
                            )
                            signal.composite_score = blended_score
                            
                            # If ML strongly disagrees, add to warnings
                            if ml_prediction.ml_signal != signal.signal:
                                if ml_prediction.ml_confidence > 0.7:
                                    signal.key_negatives.append(
                                        f"ML model suggests {ml_prediction.ml_signal} "
                                        f"({ml_prediction.ml_confidence:.0%} confidence)"
                                    )
                                    
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
                    ml_prediction = None
            
            # Generate explanations
            explain = self.explainer.generate_explanation(
                signal, profile, gov, fin, val, mkt, info,
                [{'severity': r.severity, 'description': r.description} for r in red_flags]
            )
            
            return {
                'signal': signal, 
                'explain': explain, 
                'governance': gov, 
                'financial': fin,
                'valuation': val, 
                'market': mkt, 
                'red_flags': red_flags,
                'ml_prediction': ml_prediction,  # NEW: ML results
            }


def main():
    if 'app' not in st.session_state:
        st.session_state.app = IndianEquityIntelligence()
        st.session_state.ml_init_attempted = False
    app = st.session_state.app
    
    # Auto-initialize ML models on first load if configured
    if not st.session_state.ml_init_attempted:
        ml_config = app.config.get('ml_config', {})
        if ml_config.get('enabled') and ml_config.get('auto_initialize', True):
            with st.spinner("🚀 Initializing ML models... (first time only)"):
                success, messages = app.initialize_ml()
                if success:
                    st.toast("✓ ML models ready!", icon="🤖")
                else:
                    # Don't show error, just log it
                    for msg in messages:
                        logger.info(f"ML init: {msg}")
        st.session_state.ml_init_attempted = True
    
    with st.sidebar:
        st.title("📊 Indian Equity Intelligence")
        st.caption("Local-first • Explainable • Long-term")
        st.markdown("---")
        profile_dict = render_user_profile()
        profile = UserProfile(expected_return=profile_dict['expected_return'],
                              risk_appetite=profile_dict['risk_appetite'],
                              holding_tenure=profile_dict['holding_tenure'])
        st.markdown("---")
        st.subheader("Data Status")
        try:
            available = app.data_manager.get_available_sources()
            for s in available:
                st.success(f"✓ {s.replace('_', ' ').title()}")
        except Exception:
            st.info("Data sources load on first search")
        
        # ML Models Status
        st.markdown("---")
        st.subheader("🤖 ML Models")
        ml_status = app.get_ml_status()
        
        if ml_status.get('enabled'):
            if not ml_status.get('initialized'):
                if st.button("🚀 Initialize ML Models", use_container_width=True):
                    with st.spinner("Loading ML models..."):
                        success, messages = app.initialize_ml()
                        for msg in messages:
                            st.info(msg)
                        if success:
                            st.success("ML models ready! Re-run analysis to see ML insights.")
                            # Clear previous results to force re-analysis
                            if 'results' in st.session_state:
                                del st.session_state.results
                        st.rerun()
            else:
                # Show status of each layer
                for layer in ['forecaster', 'classifier', 'explainer']:
                    layer_info = ml_status.get(layer, {})
                    model_name = layer_info.get('model') or 'Not configured'
                    is_ready = layer_info.get('ready', False)
                    
                    if is_ready:
                        st.success(f"✓ {layer.title()}: {model_name}")
                    elif model_name != 'Not configured':
                        st.warning(f"⚠ {layer.title()}: {model_name}")
                    else:
                        st.info(f"○ {layer.title()}: Disabled")
                
                # Show re-analyze button if results exist but ML wasn't used
                if hasattr(st.session_state, 'results') and st.session_state.results:
                    ml_pred = st.session_state.results.get('ml_prediction')
                    if not ml_pred or not ml_pred.success:
                        if st.button("🔄 Re-analyze with ML", use_container_width=True):
                            symbol = st.session_state.get('symbol')
                            current_profile = st.session_state.get('profile', profile)
                            if symbol:
                                data = app.fetch_data(symbol)
                                if data:
                                    st.session_state.data = data
                                    st.session_state.results = app.run_analysis(symbol, current_profile, data)
                                    st.rerun()
        else:
            st.info("ML disabled in config")
        
        render_footer()
    
    st.title("Stock Analysis")
    
    # Initialize session state for suggestions
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    
    # Load available stock list for autocomplete
    stock_list = []
    try:
        stock_list_file = Path("data/stock_lists/all_nse_stocks.json")
        if stock_list_file.exists():
            import json
            with open(stock_list_file) as f:
                stock_data = json.load(f)
                if isinstance(stock_data, list):
                    stock_list = sorted([s.get('symbol', s) if isinstance(s, dict) else s for s in stock_data[:500]])
                elif isinstance(stock_data, dict):
                    stock_list = sorted(list(stock_data.keys())[:500])
    except Exception:
        pass
    
    # Popular stocks for quick autocomplete
    popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", 
                      "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "AXISBANK",
                      "MARUTI", "TITAN", "SUNPHARMA", "HCLTECH", "WIPRO", "TECHM"]
    
    # Stock input with autocomplete
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        if stock_list:
            # Autocomplete with selectbox
            all_stocks = [""] + popular_stocks + [s for s in stock_list if s not in popular_stocks]
            default_idx = 0
            if 'selected_stock' in st.session_state and st.session_state.selected_stock in all_stocks:
                default_idx = all_stocks.index(st.session_state.selected_stock)
            symbol = st.selectbox(
                "Select or type NSE Stock Symbol",
                all_stocks,
                index=default_idx,
                format_func=lambda x: x if x else "Type to search...",
                key="stock_selector"
            )
        else:
            symbol = st.text_input("Enter NSE Stock Symbol", placeholder="e.g., RELIANCE, TCS, INFY").upper().strip()
    with col2:
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col3:
        suggest = st.button("💡 Suggest", use_container_width=True)
    with col4:
        reset = st.button("🔄 Reset", use_container_width=True)
    
    # Handle reset
    if reset:
        for key in ['results', 'data', 'symbol', 'show_suggestions', 'selected_stock']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Toggle suggestions panel
    if suggest:
        st.session_state.show_suggestions = not st.session_state.show_suggestions
    
    # Stock suggestions panel - based on user profile
    if st.session_state.show_suggestions:
        risk_appetite = profile_dict.get('risk_appetite', 'medium')
        
        # Define suggestions based on risk profile
        suggestions = {
            'low': {
                'title': '📊 Conservative Picks (Low Risk)',
                'categories': {
                    'Blue Chips': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR'],
                    'Dividend Stocks': ['ITC', 'POWERGRID', 'COALINDIA', 'ONGC', 'NTPC'],
                    'Stable Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK'],
                }
            },
            'medium': {
                'title': '⚖️ Balanced Picks (Medium Risk)',
                'categories': {
                    'Quality Growth': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
                    'Banking & Finance': ['BAJFINANCE', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI', 'MUTHOOTFIN'],
                    'Consumer': ['TITAN', 'DMART', 'TRENT', 'PAGEIND', 'BATAINDIA'],
                }
            },
            'high': {
                'title': '🚀 Growth Picks (High Risk)',
                'categories': {
                    'Mid-Cap IT': ['PERSISTENT', 'COFORGE', 'LTIM', 'MPHASIS', 'TATAELXSI'],
                    'Small-Cap Growth': ['DEEPAKNTR', 'POLYCAB', 'AFFLE', 'HAPPSTMNDS', 'ROUTE'],
                    'Emerging Sectors': ['IRCTC', 'CDSL', 'AAVAS', 'APTUS', 'DIXON'],
                }
            }
        }
        
        profile_suggestions = suggestions.get(risk_appetite, suggestions['medium'])
        
        st.info(f"**{profile_suggestions['title']}** (Based on your {risk_appetite.title()} risk profile) - Click any stock to analyze")
        
        cols = st.columns(len(profile_suggestions['categories']))
        for col, (category, stocks) in zip(cols, profile_suggestions['categories'].items()):
            with col:
                st.markdown(f"**{category}**")
                for s in stocks:
                    if st.button(s, key=f"sug_{s}", use_container_width=True):
                        # Set the selected stock and trigger analysis
                        st.session_state.selected_stock = s
                        st.session_state.analyze_stock = s
                        st.session_state.show_suggestions = False
                        st.rerun()
    
    # Handle stock selection from suggestions
    if 'analyze_stock' in st.session_state and st.session_state.analyze_stock:
        symbol = st.session_state.analyze_stock
        del st.session_state.analyze_stock
        # Auto-analyze
        st.session_state.symbol = symbol
        st.session_state.profile = profile
        with st.spinner(f"Analyzing {symbol}..."):
            data = app.fetch_data(symbol)
            if data:
                st.session_state.data = data
                st.session_state.results = app.run_analysis(symbol, profile, data)
    
    if analyze and symbol:
        st.session_state.symbol = symbol
        st.session_state.profile = profile
        data = app.fetch_data(symbol)
        if data:
            st.session_state.data = data
            st.session_state.results = app.run_analysis(symbol, profile, data)
    
    if hasattr(st.session_state, 'results') and st.session_state.results:
        results = st.session_state.results
        data = st.session_state.data
        symbol = st.session_state.symbol
        
        signal = results['signal']
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_signal_badge(signal.signal, signal.composite_score)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🤖 ML Insights", "⚠️ Why NOT", "📈 Charts", "⏰ Time Travel", "🎯 Scenarios"])
        
        with tab1:
            st.markdown(f"### {data['stock_info'].get('name', symbol)}")
            st.markdown(f"*{results['explain'].summary}*")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_score_radar(signal.dimension_scores), use_container_width=True)
            with col2:
                render_dimension_scores(signal.dimension_scores)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Key Positives")
                for p in signal.key_positives:
                    st.success(p)
            with col2:
                st.subheader("⚠️ Key Concerns")
                for n in signal.key_negatives[:5]:
                    st.warning(n)
            
            if results['red_flags']:
                st.markdown("---")
                render_red_flags([{'severity': r.severity, 'description': r.description} for r in results['red_flags']])
        
        with tab2:
            # ML Insights Tab - NEW
            st.markdown("### 🤖 ML-Enhanced Analysis")
            
            ml_prediction = results.get('ml_prediction')
            
            if ml_prediction and ml_prediction.success:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "ML Signal",
                        ml_prediction.ml_signal,
                        f"{ml_prediction.ml_confidence:.0%} confidence"
                    )
                
                with col2:
                    st.metric(
                        "ML Score",
                        f"{ml_prediction.ml_score:.0f}/100"
                    )
                
                with col3:
                    st.metric(
                        "Price Trend",
                        ml_prediction.price_trend.capitalize(),
                        f"Strength: {ml_prediction.forecast.trend_strength:.0%}" if ml_prediction.forecast else ""
                    )
                
                st.markdown("---")
                
                # Price Predictions
                if ml_prediction.price_prediction_5d or ml_prediction.price_prediction_30d:
                    st.subheader("📈 Price Predictions")
                    col1, col2 = st.columns(2)
                    
                    # Get current price from the 'close' column
                    price_df = data['price_history']
                    if 'close' in price_df.columns:
                        current_price = float(price_df['close'].iloc[-1])
                    elif 'Close' in price_df.columns:
                        current_price = float(price_df['Close'].iloc[-1])
                    else:
                        current_price = float(price_df.iloc[-1, 3])  # Fallback to 4th column
                    
                    with col1:
                        if ml_prediction.price_prediction_5d and current_price > 0:
                            change_5d = ((ml_prediction.price_prediction_5d - current_price) / current_price) * 100
                            st.metric(
                                "5-Day Prediction",
                                f"₹{ml_prediction.price_prediction_5d:.2f}",
                                f"{change_5d:+.1f}%"
                            )
                    
                    with col2:
                        if ml_prediction.price_prediction_30d and current_price > 0:
                            change_30d = ((ml_prediction.price_prediction_30d - current_price) / current_price) * 100
                            st.metric(
                                "30-Day Prediction",
                                f"₹{ml_prediction.price_prediction_30d:.2f}",
                                f"{change_30d:+.1f}%"
                            )
                    
                    st.caption("⚠️ Price predictions are probabilistic estimates, not guarantees.")
                
                st.markdown("---")
                
                # ML Summary
                if ml_prediction.summary:
                    st.subheader("📝 AI Analysis Summary")
                    st.markdown(ml_prediction.summary)
                
                # Key Insights
                if ml_prediction.key_insights:
                    st.subheader("💡 Key Insights")
                    for insight in ml_prediction.key_insights:
                        st.info(f"• {insight}")
                
                st.markdown("---")
                
                # Classifier Probabilities
                if ml_prediction.classification and ml_prediction.classification.signal_probabilities:
                    st.subheader("📊 Signal Probabilities")
                    probs = ml_prediction.classification.signal_probabilities
                    
                    for signal, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        color = {
                            "BUY": "green",
                            "HOLD": "blue", 
                            "AVOID": "orange",
                            "SELL": "red"
                        }.get(signal, "gray")
                        
                        st.progress(prob, text=f"{signal}: {prob:.0%}")
                
                # Performance info
                st.markdown("---")
                st.caption(
                    f"Layers used: {', '.join(ml_prediction.layers_used)} | "
                    f"Inference time: {ml_prediction.total_inference_time_ms:.0f}ms"
                )
                
                if ml_prediction.errors:
                    with st.expander("⚠️ Warnings"):
                        for error in ml_prediction.errors:
                            st.warning(error)
            else:
                # Check ML status to give specific guidance
                ml_status = app.get_ml_status()
                
                if not ml_status.get('enabled'):
                    st.warning("**ML is disabled in configuration.**")
                    st.info("Enable ML in `config/settings.yaml` by setting `ml_config.enabled: true`")
                    
                elif not ml_status.get('initialized'):
                    st.warning("**ML models not initialized.**")
                    st.info("👈 Click **'Initialize ML Models'** in the sidebar, then re-analyze.")
                    
                else:
                    # ML is initialized but prediction failed
                    st.warning("**ML analysis ran but didn't produce results.**")
                    
                    # Show which layers are ready
                    st.markdown("**Layer Status:**")
                    for layer in ['forecaster', 'classifier', 'explainer']:
                        layer_info = ml_status.get(layer, {})
                        is_ready = layer_info.get('ready', False)
                        model_name = layer_info.get('model', 'Not configured')
                        
                        if is_ready:
                            st.success(f"✓ {layer.title()}: {model_name}")
                        elif model_name:
                            st.warning(f"⚠ {layer.title()}: {model_name} (not ready)")
                        else:
                            st.info(f"○ {layer.title()}: Disabled")
                    
                    # Check if classifier needs training
                    classifier_info = ml_status.get('classifier', {})
                    if classifier_info.get('model') and not classifier_info.get('ready'):
                        st.info("""
                        💡 **Tip**: The classifier needs training data. It will improve as you analyze more stocks.
                        For now, forecaster and explainer results are still available.
                        """)
                    
                    if st.button("🔄 Re-run ML Analysis", use_container_width=True):
                        symbol = st.session_state.get('symbol')
                        current_profile = st.session_state.get('profile')
                        if symbol and current_profile:
                            data = app.fetch_data(symbol)
                            if data:
                                st.session_state.data = data
                                st.session_state.results = app.run_analysis(symbol, current_profile, data)
                                st.rerun()
                
                # Show download instructions
                with st.expander("📥 Model Download Instructions"):
                    st.markdown("""
                    ### Quick Start
                    
                    ```bash
                    # Install ML dependencies
                    pip install -r requirements.txt
                    
                    # Download Chronos-T5-Base (forecaster)
                    huggingface-cli download amazon/chronos-t5-base --local-dir models/forecaster/chronos-t5-base
                    
                    # Download Qwen2.5-3B (explainer) - GGUF format
                    huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF --local-dir models/explainer/qwen2.5-3b --include "*.gguf"
                    ```
                    
                    See ML_MODELS.md for complete instructions.
                    """)
        
        with tab3:
            st.markdown("### This section is shown for ALL stocks, even BUY signals")
            st.markdown("---")
            render_why_not_buy(results['explain'].why_not_buy)
            st.markdown("---")
            st.subheader("Risk Factors")
            for r in results['explain'].risk_factors:
                st.error(f"⚠️ {r}")
            st.markdown("---")
            st.subheader("What Could Invalidate The Thesis")
            for i in results['explain'].thesis_invalidators:
                st.info(f"📌 {i}")
        
        with tab4:
            st.subheader("Price History")
            st.plotly_chart(create_price_chart(data['price_history'], f"{symbol} Price"), use_container_width=True)
            st.subheader("Drawdown History")
            st.plotly_chart(create_drawdown_chart(data['price_history']), use_container_width=True)
        
        with tab5:
            st.subheader("⏰ Time Travel Mode")
            st.info("Re-analyze using only data available at a historical point. No future leakage.")
            cutoffs = app.time_travel.get_available_cutoffs(data['price_history'])
            if cutoffs:
                cutoff = st.selectbox("Travel back to December of:", cutoffs)
                if st.button("🕰️ Run Time Travel", type="primary"):
                    with st.spinner(f"Traveling back to {cutoff}..."):
                        try:
                            tt_result = app.time_travel.analyze_at_cutoff(
                                symbol=symbol,
                                cutoff_year=cutoff,
                                user_profile=profile,
                                stock_info=data['stock_info'],
                                price_history=data['price_history'],
                                financials=data['financials'],
                                shareholding=data.get('shareholding', pd.DataFrame()),
                                dividends=data.get('dividends', pd.DataFrame()),
                                nifty_history=data.get('nifty_history'),
                                include_outcome=True
                            )
                            
                            st.success(f"Analysis complete for December {cutoff}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"### Signal at {cutoff}")
                                render_signal_badge(tt_result.signal_at_cutoff.signal, tt_result.signal_at_cutoff.composite_score)
                            
                            with col2:
                                st.markdown("### Actual Outcome")
                                if tt_result.actual_outcome:
                                    for period, data_out in tt_result.actual_outcome.items():
                                        if isinstance(data_out, dict) and 'return' in data_out:
                                            ret = data_out['return']
                                            color = "green" if ret > 0 else "red"
                                            st.markdown(f"**{period.upper()}**: <span style='color:{color}'>{ret:+.1f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.warning("Outcome data not available")
                            
                            st.markdown("---")
                            st.markdown("### Hindsight Analysis")
                            st.info(tt_result.hindsight_analysis)
                            
                            # Show red flags at that time
                            if tt_result.red_flags_at_cutoff:
                                st.markdown("### Red Flags (at that time)")
                                for flag in tt_result.red_flags_at_cutoff:
                                    st.warning(f"⚠️ {flag.get('description', flag)}")
                                    
                        except Exception as e:
                            st.error(f"Time Travel failed: {e}")
            else:
                st.warning("Not enough historical data for time travel.")
        
        with tab6:
            st.subheader("🎯 Scenario Simulator")
            st.info("Stress-test your investment thesis under various market conditions")
            
            scenarios = app.scenario_sim.list_scenarios()
            selected = st.selectbox("Select Scenario:", [s['key'] for s in scenarios],
                                    format_func=lambda x: next(s['name'] for s in scenarios if s['key'] == x))
            
            for s in scenarios:
                if s['key'] == selected:
                    st.caption(s['description'])
            
            if st.button("🎯 Run Scenario", type="primary"):
                with st.spinner("Running scenario simulation..."):
                    try:
                        # Build financial metrics and valuation from results
                        # Get operating margin from details dict or use default
                        fin_details = results['financial'].details or {}
                        opm = fin_details.get('opm_current', fin_details.get('operating_margin', 15))
                        
                        fin_metrics = {
                            'revenue_cagr_5y': results['financial'].revenue_cagr_5y or 10,
                            'operating_margin': opm,
                        }
                        # Get current price from price history
                        price_df = data['price_history']
                        if len(price_df) > 0:
                            if 'close' in price_df.columns:
                                curr_price = float(price_df['close'].iloc[-1])
                            elif 'Close' in price_df.columns:
                                curr_price = float(price_df['Close'].iloc[-1])
                            else:
                                curr_price = 100
                        else:
                            curr_price = 100
                        
                        val_metrics = {
                            'pe_ratio': results['valuation'].pe_ratio or 20,
                            'current_price': curr_price,
                        }
                        user_profile_dict = {
                            'holding_tenure': profile.holding_tenure,
                            'expected_return': profile.expected_return,
                        }
                        
                        scenario_result = app.scenario_sim.simulate(
                            scenario_name=selected,
                            stock_info=data['stock_info'],
                            financial_metrics=fin_metrics,
                            valuation=val_metrics,
                            user_profile=user_profile_dict
                        )
                        
                        st.success(f"Scenario: {scenario_result.scenario.name}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Base Case")
                            st.metric("Expected Return", f"{scenario_result.base_case['expected_return']:.1f}%")
                            st.metric("Target Price", f"₹{scenario_result.base_case['target_price']:.0f}")
                        
                        with col2:
                            st.markdown("### Stressed Case")
                            stressed_ret = scenario_result.stressed_case['expected_return']
                            base_ret = scenario_result.base_case['expected_return']
                            st.metric("Expected Return", f"{stressed_ret:.1f}%", f"{stressed_ret - base_ret:.1f}%")
                            st.metric("Target Price", f"₹{scenario_result.stressed_case['target_price']:.0f}")
                        
                        st.markdown("---")
                        
                        # Resilience rating
                        resilience_colors = {'robust': 'green', 'resilient': 'blue', 'moderate': 'orange', 'fragile': 'red'}
                        color = resilience_colors.get(scenario_result.resilience_rating, 'gray')
                        st.markdown(f"### Resilience Rating: <span style='color:{color}'>{scenario_result.resilience_rating.upper()}</span>", unsafe_allow_html=True)
                        
                        if scenario_result.signal_change:
                            st.warning(f"⚠️ {scenario_result.signal_change}")
                        
                        st.markdown("### Key Findings")
                        for finding in scenario_result.key_findings:
                            st.info(f"• {finding}")
                            
                    except Exception as e:
                        st.error(f"Scenario simulation failed: {e}")
    
    else:
        st.markdown("""
            ## Welcome to Indian Equity Intelligence
            
            A **local-first**, **explainable** tool for long-term stock analysis.
            
            ### How to Use
            1. Enter a stock symbol (e.g., RELIANCE, TCS, INFY)
            2. Set your investment profile in the sidebar
            3. Click **Analyze**
            
            ### Core Philosophy
            > *"Is this stock suitable for this investor, under these assumptions — 
            > and what could invalidate the thesis?"*
            
            ### Features
            - **Contextual Signals**: BUY, HOLD, AVOID, SELL based on YOUR profile
            - **"Why NOT to Buy"**: Shown even for BUY signals
            - **Red Flag Detection**: Governance and financial warnings
            - **Time Travel Mode**: Re-analyze with historical data
            - **Scenario Simulation**: Stress-test assumptions
            
            ### Privacy
            - 100% local processing after data download
            - No data sent to external servers
        """)


if __name__ == "__main__":
    main()
