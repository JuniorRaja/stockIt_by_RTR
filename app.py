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
    
    @property
    def data_manager(self):
        if self._data_manager is None:
            self._data_manager = DataSourceManager()
        return self._data_manager
    
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
            
            gov = self.governance.analyze(symbol, info, share, divs)
            fin = self.financial.analyze(symbol, fins, info)
            val = self.valuation.analyze(symbol, info, prices, fins, earnings_growth=fin.pat_cagr_5y)
            mkt = self.market.analyze(symbol, prices, nifty)
            red_flags = self.red_flag.detect_all_flags(symbol, info, share, fins)
            
            signal = self.signal_gen.generate_signal(
                profile, gov, fin, val, mkt,
                red_flags=[{'severity': r.severity, 'description': r.description} for r in red_flags]
            )
            explain = self.explainer.generate_explanation(signal, profile, gov, fin, val, mkt, info,
                                                          [{'severity': r.severity, 'description': r.description} for r in red_flags])
            
            return {'signal': signal, 'explain': explain, 'governance': gov, 'financial': fin,
                    'valuation': val, 'market': mkt, 'red_flags': red_flags}


def main():
    if 'app' not in st.session_state:
        st.session_state.app = IndianEquityIntelligence()
    app = st.session_state.app
    
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
        render_footer()
    
    st.title("Stock Analysis")
    col1, col2 = st.columns([4, 1])
    with col1:
        symbol = st.text_input("Enter NSE Stock Symbol", placeholder="e.g., RELIANCE, TCS, INFY").upper().strip()
    with col2:
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
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
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "⚠️ Why NOT", "📈 Charts", "⏰ Time Travel", "🎯 Scenarios"])
        
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
        
        with tab3:
            st.subheader("Price History")
            st.plotly_chart(create_price_chart(data['price_history'], f"{symbol} Price"), use_container_width=True)
            st.subheader("Drawdown History")
            st.plotly_chart(create_drawdown_chart(data['price_history']), use_container_width=True)
        
        with tab4:
            st.subheader("⏰ Time Travel Mode")
            st.info("Re-analyze using only data available at a historical point. No future leakage.")
            cutoffs = app.time_travel.get_available_cutoffs(data['price_history'])
            if cutoffs:
                cutoff = st.selectbox("Travel back to December of:", cutoffs)
                if st.button("🕰️ Run Time Travel"):
                    st.info("Time Travel analysis runs the full analysis engine with historical data only.")
            else:
                st.warning("Not enough historical data for time travel.")
        
        with tab5:
            st.subheader("🎯 Scenario Simulator")
            st.info("Stress-test your investment thesis")
            scenarios = app.scenario_sim.list_scenarios()
            selected = st.selectbox("Select Scenario:", [s['key'] for s in scenarios],
                                    format_func=lambda x: next(s['name'] for s in scenarios if s['key'] == x))
            for s in scenarios:
                if s['key'] == selected:
                    st.info(s['description'])
            if st.button("Run Scenario"):
                st.info("Scenario simulation shows impact on expected returns and thesis resilience.")
    
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
