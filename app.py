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
import json

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

# Page config
st.set_page_config(
    page_title="Indian Equity Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 0 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div[data-testid="metric-container"] { 
        background-color: #262730; 
        padding: 10px; 
        border-radius: 5px; 
    }
    .resilience-robust { color: #00ff00; }
    .resilience-resilient { color: #90EE90; }
    .resilience-moderate { color: #FFA500; }
    .resilience-fragile { color: #FF6347; }
</style>
""", unsafe_allow_html=True)


class IndianEquityIntelligence:
    """Main application class."""
    
    def __init__(self):
        self.config = load_config()
        self._data_manager = None
        self._db_manager = None
        self._stock_list = None
        
        # Lazy-loaded analyzers
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
    
    def get_stock_list(self) -> pd.DataFrame:
        """Get cached stock list."""
        if self._stock_list is not None:
            return self._stock_list
        
        stock_list_path = Path(__file__).parent / 'data' / 'stock_lists' / 'all_stocks.json'
        if stock_list_path.exists():
            try:
                with open(stock_list_path, 'r') as f:
                    data = json.load(f)
                    self._stock_list = pd.DataFrame(data['stocks'])
                    return self._stock_list
            except Exception:
                pass
        
        self._stock_list = self.data_manager.get_all_stocks()
        return self._stock_list
    
    def search_stocks(self, query: str) -> pd.DataFrame:
        """Search stocks by symbol or name (min 2 characters)."""
        if len(query) < 2:
            return pd.DataFrame()
        
        stocks = self.get_stock_list()
        if stocks.empty:
            return pd.DataFrame()
        
        query = query.upper()
        symbol_match = stocks[stocks['symbol'].str.upper().str.startswith(query)]
        name_match = stocks[~stocks['symbol'].isin(symbol_match['symbol']) & 
                           stocks['name'].str.upper().str.contains(query, na=False)]
        
        result = pd.concat([symbol_match, name_match]).head(20)
        return result
    
    def get_stock_suggestions(self, risk_level: str, min_price: float = 0, 
                               max_price: float = float('inf')) -> list:
        """Get stock suggestions based on risk profile and price filter."""
        low_risk = ['HDFCBANK', 'TCS', 'INFY', 'HINDUNILVR', 'NESTLEIND', 
                    'BRITANNIA', 'PIDILITIND', 'DABUR', 'MARICO', 'ITC']
        medium_risk = ['RELIANCE', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK',
                       'LT', 'ASIANPAINT', 'TITAN', 'BAJFINANCE', 'MARUTI']
        high_risk = ['TATAMOTORS', 'VEDL', 'TATAPOWER', 'ADANIENT', 'ADANIPORTS',
                     'ZOMATO', 'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY']
        
        if risk_level.lower() == 'low':
            pool = low_risk
        elif risk_level.lower() == 'high':
            pool = high_risk
        else:
            pool = medium_risk
        
        suggestions = []
        for symbol in pool:
            try:
                info = self.data_manager.get_stock_info(symbol, use_cache=True)
                if info:
                    price = info.get('current_price', 0) or 0
                    if min_price <= price <= max_price:
                        suggestions.append({
                            'symbol': symbol,
                            'name': info.get('name', symbol),
                            'price': price,
                            'pe': info.get('pe_ratio'),
                            'sector': info.get('sector', 'Unknown')
                        })
            except Exception:
                pass
        
        return suggestions[:10]
    
    def fetch_data(self, symbol: str):
        """Fetch all data for a symbol."""
        with st.spinner(f"Fetching data for {symbol}..."):
            info = self.data_manager.get_stock_info(symbol)
            if not info:
                return None, "Could not find stock information"
            
            prices = self.data_manager.get_price_history(symbol, years=10)
            if prices is None or prices.empty:
                prices = self.data_manager.get_price_history(symbol, years=5)
            
            if prices is None or prices.empty:
                return None, f"No price history available for {symbol}. Try refreshing data."
            
            financials = self.data_manager.get_financials(symbol) or {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }
            
            nifty = self.data_manager.get_price_history("NIFTY50", years=10)
            if nifty is None:
                nifty = self.data_manager.get_price_history("^NSEI", years=10)
            
            return {
                'stock_info': info,
                'price_history': prices,
                'financials': financials,
                'nifty_history': nifty,
                'shareholding': pd.DataFrame(),
                'dividends': pd.DataFrame()
            }, None
    
    def run_analysis(self, symbol: str, profile: UserProfile, data: dict):
        """Run complete analysis."""
        with st.spinner("Running multi-dimensional analysis..."):
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
                'red_flags': red_flags
            }
    
    def run_time_travel(self, symbol: str, cutoff_year: int, profile: UserProfile, data: dict):
        """Run time travel analysis."""
        return self.time_travel.analyze_at_cutoff(
            symbol=symbol,
            cutoff_year=cutoff_year,
            user_profile=profile,
            stock_info=data['stock_info'],
            price_history=data['price_history'],
            financials=data['financials'],
            shareholding=data.get('shareholding', pd.DataFrame()),
            dividends=data.get('dividends', pd.DataFrame()),
            nifty_history=data.get('nifty_history'),
            include_outcome=True
        )
    
    def run_scenario(self, scenario_name: str, data: dict, results: dict, profile: dict):
        """Run scenario simulation."""
        financial_metrics = {
            'revenue_cagr_5y': results['financial'].revenue_cagr_5y or 10,
            'operating_margin': 15,  # Default estimate
        }
        valuation = {
            'pe_ratio': data['stock_info'].get('pe_ratio') or 20,
            'current_price': data['stock_info'].get('current_price') or 100,
        }
        return self.scenario_sim.simulate(scenario_name, data['stock_info'], 
                                           financial_metrics, valuation, profile)


def render_stock_search(app):
    """Render stock search with autocomplete."""
    st.subheader("🔍 Search Stock")
    
    search_query = st.text_input(
        "Enter stock symbol or name (min 2 characters)",
        placeholder="e.g., RELIANCE, TCS, Infosys...",
        key="stock_search"
    ).strip()
    
    selected_symbol = None
    
    if len(search_query) >= 2:
        matches = app.search_stocks(search_query)
        if not matches.empty:
            st.caption(f"Found {len(matches)} matches:")
            cols = st.columns(4)
            for idx, row in matches.iterrows():
                col_idx = idx % 4
                with cols[col_idx]:
                    if st.button(
                        f"**{row['symbol']}**\n{row['name'][:20]}...",
                        key=f"stock_{row['symbol']}",
                        use_container_width=True
                    ):
                        selected_symbol = row['symbol']
                        st.session_state.selected_symbol = selected_symbol
        else:
            st.info("No stocks found matching your search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_symbol = st.text_input(
            "Or enter symbol directly",
            value=st.session_state.get('selected_symbol', ''),
            placeholder="SYMBOL"
        ).upper().strip()
    with col2:
        st.write("")
        st.write("")
        analyze_clicked = st.button("📊 Analyze", type="primary", use_container_width=True)
    
    if analyze_clicked and manual_symbol:
        return manual_symbol
    
    return selected_symbol


def render_suggestions(app, risk_level: str, min_price: float, max_price: float, key_prefix: str = "main"):
    """Render stock suggestions based on profile."""
    st.subheader("💡 Stock Suggestions")
    st.caption(f"Based on {risk_level} risk profile, price ₹{min_price:.0f} - ₹{max_price:.0f}")
    
    with st.spinner("Loading suggestions..."):
        suggestions = app.get_stock_suggestions(risk_level, min_price, max_price)
    
    if suggestions:
        for idx, sugg in enumerate(suggestions):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{sugg['symbol']}** - {sugg['name'][:25]}...")
            with col2:
                st.markdown(f"₹{sugg['price']:.2f}" if sugg['price'] else "N/A")
            with col3:
                if st.button("Analyze", key=f"sugg_{key_prefix}_{idx}_{sugg['symbol']}"):
                    st.session_state.selected_symbol = sugg['symbol']
                    st.session_state.trigger_analysis = True
                    st.rerun()
    else:
        st.info("No suggestions available. Try adjusting price range.")


def render_portfolio_upload():
    """Render portfolio upload section."""
    st.subheader("📁 Upload Portfolio")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with columns: symbol, quantity, buy_price (optional)",
        type=['csv'],
        key="portfolio_upload"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['symbol']
            if not all(col in df.columns for col in required_cols):
                st.error("CSV must have 'symbol' column")
                return None
            
            st.success(f"Loaded {len(df)} stocks from portfolio")
            st.dataframe(df.head(10))
            
            # Save to session state
            st.session_state.portfolio = df
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Show sample format
    with st.expander("📋 Sample CSV Format"):
        st.code("""symbol,quantity,buy_price
RELIANCE,10,2500
TCS,5,3800
INFY,20,1500
HDFCBANK,15,1650""")
    
    return st.session_state.get('portfolio')


def render_time_travel_results(tt_result):
    """Render time travel analysis results."""
    st.markdown(f"### Analysis as of December {tt_result.cutoff_date.year}")
    
    # Signal at cutoff
    col1, col2 = st.columns(2)
    with col1:
        render_signal_badge(tt_result.signal_at_cutoff.signal, tt_result.signal_at_cutoff.composite_score)
    with col2:
        st.metric("Score at Cutoff", f"{tt_result.signal_at_cutoff.composite_score:.0f}/100")
    
    # Actual outcome
    if tt_result.actual_outcome:
        st.markdown("### 📈 What Actually Happened")
        outcome = tt_result.actual_outcome
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Price at Cutoff", f"₹{outcome.get('price_at_cutoff', 0):.2f}")
        
        for i, period in enumerate(['1y', '3y', '5y']):
            if period in outcome:
                ret = outcome[period].get('return', 0)
                with cols[i+1]:
                    st.metric(f"{period} Return", f"{ret:+.1f}%", 
                             delta_color="normal" if ret > 0 else "inverse")
    
    # Hindsight analysis
    if tt_result.hindsight_analysis:
        st.markdown("### 🔮 Hindsight Analysis")
        if "CORRECT" in tt_result.hindsight_analysis:
            st.success(tt_result.hindsight_analysis)
        elif "INCORRECT" in tt_result.hindsight_analysis:
            st.error(tt_result.hindsight_analysis)
        else:
            st.info(tt_result.hindsight_analysis)
    
    # Red flags at cutoff
    if tt_result.red_flags_at_cutoff:
        st.markdown("### 🚩 Red Flags at Cutoff")
        for rf in tt_result.red_flags_at_cutoff:
            st.warning(f"{rf['severity'].upper()}: {rf['description']}")


def render_scenario_results(scenario_result):
    """Render scenario simulation results."""
    st.markdown(f"### {scenario_result.scenario.name}")
    st.caption(scenario_result.scenario.description)
    
    # Impact metrics
    col1, col2, col3, col4 = st.columns(4)
    impact = scenario_result.impact_summary
    
    with col1:
        st.metric("Return Impact", f"{impact['return_impact']:+.1f}%",
                 delta_color="inverse" if impact['return_impact'] < 0 else "normal")
    with col2:
        st.metric("Margin Impact", f"{impact['margin_impact']:+.1f}pp",
                 delta_color="inverse" if impact['margin_impact'] < 0 else "normal")
    with col3:
        st.metric("Target Price Impact", f"{impact['target_impact_pct']:+.1f}%",
                 delta_color="inverse" if impact['target_impact_pct'] < 0 else "normal")
    with col4:
        resilience_color = {
            'robust': '🟢', 'resilient': '🟡', 'moderate': '🟠', 'fragile': '🔴'
        }.get(scenario_result.resilience_rating, '⚪')
        st.metric("Resilience", f"{resilience_color} {scenario_result.resilience_rating.title()}")
    
    # Signal change
    if scenario_result.signal_change:
        st.warning(f"⚠️ Signal Change: {scenario_result.signal_change}")
    
    # Key findings
    st.markdown("**Key Findings:**")
    for finding in scenario_result.key_findings:
        st.markdown(f"• {finding}")
    
    # Base vs Stressed comparison
    with st.expander("📊 Base Case vs Stressed Case"):
        base = scenario_result.base_case
        stressed = scenario_result.stressed_case
        
        compare_df = pd.DataFrame({
            'Metric': ['Revenue Growth', 'Operating Margin', 'Expected Return', 'Target Price'],
            'Base Case': [f"{base['rev_growth']:.1f}%", f"{base['opm']:.1f}%", 
                         f"{base['expected_return']:.1f}%", f"₹{base['target_price']:.0f}"],
            'Stressed Case': [f"{stressed['rev_growth']:.1f}%", f"{stressed['opm']:.1f}%",
                             f"{stressed['expected_return']:.1f}%", f"₹{stressed['target_price']:.0f}"]
        })
        st.table(compare_df)


def reset_session():
    """Reset all session state."""
    keys_to_keep = ['app']  # Keep the app instance
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]


def main():
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = IndianEquityIntelligence()
    app = st.session_state.app
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Indian Equity Intelligence")
        st.caption("Local-first • Explainable • Long-term")
        
        # Reset button at top
        if st.button("🔄 Reset UI", use_container_width=True):
            reset_session()
            st.rerun()
        
        st.markdown("---")
        
        # User Profile
        profile_dict = render_user_profile()
        profile = UserProfile(
            expected_return=profile_dict['expected_return'],
            risk_appetite=profile_dict['risk_appetite'],
            holding_tenure=profile_dict['holding_tenure']
        )
        
        st.markdown("---")
        
        # Price filter
        st.subheader("📌 Stock Price Filter")
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            min_price = st.number_input("Min ₹", min_value=0, value=0, step=100)
        with price_col2:
            max_price = st.number_input("Max ₹", min_value=0, value=10000, step=100)
        
        # Get Suggestions button
        if st.button("💡 Get Suggestions", use_container_width=True, type="secondary"):
            st.session_state.show_suggestions = True
        
        st.markdown("---")
        
        # Portfolio upload
        portfolio = render_portfolio_upload()
        
        st.markdown("---")
        
        # Data Management
        st.subheader("🔄 Data Management")
        if st.button("Refresh Stock List", use_container_width=True):
            with st.spinner("Refreshing..."):
                app._stock_list = None
                app.data_manager.get_all_stocks(force_refresh=True)
                st.success("Stock list refreshed!")
        
        # Data sources
        st.caption("Data Sources:")
        try:
            available = app.data_manager.get_available_sources()
            for s in available:
                st.success(f"✓ {s.replace('_', ' ').title()}")
            if not available:
                st.warning("No sources available")
        except Exception:
            st.info("Sources load on first search")
        
        render_footer()
    
    # Main content
    st.title("Stock Analysis")
    
    # Show suggestions panel if requested
    if st.session_state.get('show_suggestions'):
        render_suggestions(app, profile_dict['risk_appetite'], min_price, max_price, key_prefix="sidebar")
        if st.button("Hide Suggestions"):
            st.session_state.show_suggestions = False
            st.rerun()
        st.markdown("---")
    
    # Stock search
    symbol = render_stock_search(app)
    
    # Check for triggered analysis from suggestions
    if st.session_state.get('trigger_analysis') and st.session_state.get('selected_symbol'):
        symbol = st.session_state.selected_symbol
        st.session_state.trigger_analysis = False
    
    # Handle analysis
    if symbol:
        st.session_state.current_symbol = symbol
        st.session_state.profile = profile
        
        data, error = app.fetch_data(symbol)
        
        if error:
            st.error(f"❌ {error}")
            if st.button("🔄 Try Refreshing Data"):
                success, msg = app.data_manager.refresh_data(symbol)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.session_state.data = data
            st.session_state.results = app.run_analysis(symbol, profile, data)
    
    # Display results
    if hasattr(st.session_state, 'results') and st.session_state.results:
        results = st.session_state.results
        data = st.session_state.data
        symbol = st.session_state.current_symbol
        profile = st.session_state.profile
        
        signal = results['signal']
        
        # Header
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"## {data['stock_info'].get('name', symbol)}")
        with col2:
            if st.button("🔄 Refresh"):
                success, msg = app.data_manager.refresh_data(symbol)
                if success:
                    data, _ = app.fetch_data(symbol)
                    if data:
                        st.session_state.data = data
                        st.session_state.results = app.run_analysis(symbol, profile, data)
                        st.rerun()
        with col3:
            if st.button("🆕 New Analysis"):
                reset_session()
                st.rerun()
        
        # Signal badge
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_signal_badge(signal.signal, signal.composite_score)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "⚠️ Why NOT", "📈 Charts", 
            "⏰ Time Travel", "🎯 Scenarios", "📁 Portfolio"
        ])
        
        with tab1:
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
                render_red_flags([
                    {'severity': r.severity, 'description': r.description} 
                    for r in results['red_flags']
                ])
        
        with tab2:
            st.markdown("### This section is shown for ALL stocks, even BUY signals")
            st.markdown("Every investment has risks. Here's what to consider:")
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
            st.info("Re-analyze using only data available at a historical point. See what signal would have been generated and what actually happened.")
            
            cutoffs = app.time_travel.get_available_cutoffs(data['price_history'])
            if cutoffs:
                col1, col2 = st.columns([2, 1])
                with col1:
                    cutoff_year = st.selectbox("Select year to travel back to:", cutoffs, key="tt_year")
                with col2:
                    st.write("")
                    run_tt = st.button("🕰️ Run Time Travel", type="primary", use_container_width=True)
                
                if run_tt:
                    with st.spinner(f"Analyzing as of December {cutoff_year}..."):
                        tt_result = app.run_time_travel(symbol, cutoff_year, profile, data)
                        st.session_state.tt_result = tt_result
                
                if st.session_state.get('tt_result'):
                    render_time_travel_results(st.session_state.tt_result)
            else:
                st.warning("Not enough historical data for time travel analysis. Need data from before 2022.")
        
        with tab5:
            st.subheader("🎯 Scenario Simulator")
            st.info("Stress-test your investment thesis under various economic scenarios.")
            
            scenarios = app.scenario_sim.list_scenarios()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_scenario = st.selectbox(
                    "Select Scenario:",
                    [s['key'] for s in scenarios],
                    format_func=lambda x: next(s['name'] for s in scenarios if s['key'] == x),
                    key="scenario_select"
                )
            with col2:
                st.write("")
                run_scenario = st.button("▶️ Run Scenario", type="primary", use_container_width=True)
            
            # Show scenario description
            for s in scenarios:
                if s['key'] == selected_scenario:
                    st.caption(f"📋 {s['description']}")
            
            if run_scenario:
                with st.spinner("Running scenario simulation..."):
                    scenario_result = app.run_scenario(
                        selected_scenario, data, results,
                        {'holding_tenure': profile.holding_tenure}
                    )
                    st.session_state.scenario_result = scenario_result
            
            if st.session_state.get('scenario_result'):
                st.markdown("---")
                render_scenario_results(st.session_state.scenario_result)
            
            # Run all scenarios
            st.markdown("---")
            if st.button("📊 Run All Scenarios"):
                with st.spinner("Running all scenario simulations..."):
                    all_results = {}
                    for s in scenarios:
                        try:
                            all_results[s['key']] = app.run_scenario(
                                s['key'], data, results,
                                {'holding_tenure': profile.holding_tenure}
                            )
                        except Exception as e:
                            st.warning(f"Scenario {s['name']} failed: {e}")
                    
                    st.session_state.all_scenarios = all_results
            
            if st.session_state.get('all_scenarios'):
                st.subheader("📊 All Scenarios Summary")
                
                summary = app.scenario_sim.get_resilience_summary(st.session_state.all_scenarios)
                st.metric("Overall Resilience", summary['overall'].title())
                
                # Summary table
                summary_data = []
                for key, result in st.session_state.all_scenarios.items():
                    summary_data.append({
                        'Scenario': result.scenario.name,
                        'Return Impact': f"{result.impact_summary['return_impact']:+.1f}%",
                        'Resilience': result.resilience_rating.title(),
                        'Signal Change': result.signal_change or 'None'
                    })
                
                st.table(pd.DataFrame(summary_data))
        
        with tab6:
            st.subheader("📁 Portfolio Analysis")
            
            portfolio = st.session_state.get('portfolio')
            
            if portfolio is not None and not portfolio.empty:
                st.success(f"Portfolio loaded with {len(portfolio)} stocks")
                
                # Show portfolio
                st.dataframe(portfolio)
                
                # Analyze portfolio button
                if st.button("📊 Analyze Portfolio"):
                    st.info("Portfolio analysis will analyze each stock in your portfolio and provide an overall assessment.")
                    
                    progress = st.progress(0)
                    results_list = []
                    
                    for idx, row in portfolio.iterrows():
                        sym = row['symbol']
                        progress.progress((idx + 1) / len(portfolio))
                        
                        try:
                            data_temp, err = app.fetch_data(sym)
                            if data_temp:
                                result = app.run_analysis(sym, profile, data_temp)
                                results_list.append({
                                    'symbol': sym,
                                    'signal': result['signal'].signal,
                                    'score': result['signal'].composite_score,
                                    'red_flags': len(result['red_flags'])
                                })
                        except Exception:
                            results_list.append({
                                'symbol': sym,
                                'signal': 'Error',
                                'score': 0,
                                'red_flags': 0
                            })
                    
                    st.session_state.portfolio_results = results_list
                
                if st.session_state.get('portfolio_results'):
                    st.subheader("Portfolio Analysis Results")
                    results_df = pd.DataFrame(st.session_state.portfolio_results)
                    st.dataframe(results_df)
                    
                    # Summary
                    buy_count = len(results_df[results_df['signal'] == 'BUY'])
                    hold_count = len(results_df[results_df['signal'] == 'HOLD'])
                    avoid_count = len(results_df[results_df['signal'].isin(['AVOID BUYING', 'SELL / EXIT'])])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BUY Signals", buy_count)
                    with col2:
                        st.metric("HOLD Signals", hold_count)
                    with col3:
                        st.metric("AVOID/SELL Signals", avoid_count)
            else:
                st.info("Upload a portfolio CSV in the sidebar to analyze multiple stocks at once.")
                
                # Sample portfolio
                st.markdown("### Try Sample Portfolio")
                if st.button("Load Sample Portfolio"):
                    sample = pd.DataFrame({
                        'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC'],
                        'quantity': [10, 5, 20, 15, 50],
                        'buy_price': [2500, 3800, 1500, 1650, 450]
                    })
                    st.session_state.portfolio = sample
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
            ## Welcome to Indian Equity Intelligence
            
            A **local-first**, **explainable** tool for long-term Indian stock analysis.
            
            ### How to Use
            1. **Search**: Type at least 2 characters to find stocks
            2. **Profile**: Set your investment profile in the sidebar
            3. **Analyze**: Click on any stock to run full analysis
            
            ---
        """)
        
        # Show suggestions on homepage
        render_suggestions(app, profile_dict['risk_appetite'], min_price, max_price, key_prefix="home")
        
        st.markdown("""
            ---
            
            ### Features
            - **Contextual Signals**: BUY, HOLD, AVOID, SELL based on YOUR profile
            - **"Why NOT to Buy"**: Shown even for BUY signals
            - **Time Travel**: See what the signal would have been in the past
            - **Scenario Simulation**: Stress-test under various scenarios
            - **Portfolio Analysis**: Analyze multiple stocks at once
        """)


if __name__ == "__main__":
    main()
