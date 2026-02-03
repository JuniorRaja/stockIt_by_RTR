"""Streamlit UI Components for Stocron by RTR."""

import streamlit as st
from typing import Dict, Any, Optional, List, Callable


def render_stock_search(on_search: Callable[[str], None], stock_list: Optional[List[str]] = None) -> Optional[str]:
    """Render stock search component."""
    st.subheader("Search Stock")
    col1, col2 = st.columns([3, 1])
    with col1:
        if stock_list:
            symbol = st.selectbox("Select stock", [""] + stock_list, format_func=lambda x: x or "Type to search...")
        else:
            symbol = st.text_input("Enter NSE Symbol", placeholder="e.g., RELIANCE, TCS, INFY").upper()
    with col2:
        clicked = st.button("Analyze", type="primary", use_container_width=True)
    if clicked and symbol:
        on_search(symbol)
        return symbol
    return None


def render_user_profile() -> Dict[str, Any]:
    """Render user profile input."""
    st.subheader("📋 Your Investment Profile")
    
    expected_return = st.slider(
        "Expected CAGR %", 
        min_value=0, max_value=50, value=15, 
        help="Your expected annual return (0% = capital preservation, 50% = very aggressive)"
    )
    
    risk = st.selectbox(
        "Risk Appetite", 
        ["Low", "Medium", "High"], 
        index=1,
        help="Low: Stable blue chips, Medium: Quality growth, High: Aggressive growth"
    )
    
    tenure = st.slider(
        "Holding Period (Years)", 
        min_value=1, max_value=60, value=5,
        help="How long you plan to hold"
    )
    
    # Price range filter
    st.markdown("---")
    st.subheader("💰 Stock Price Filter")
    
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input(
            "Min Price (₹)",
            min_value=1,
            max_value=100000,
            value=1,
            step=10,
            help="Minimum stock price"
        )
    with col2:
        max_price = st.number_input(
            "Max Price (₹)",
            min_value=1,
            max_value=100000,
            value=10000,
            step=100,
            help="Maximum stock price"
        )
    
    market_cap = st.selectbox(
        "Market Cap",
        ["Any", "Large Cap (>₹20,000 Cr)", "Mid Cap (₹5,000-20,000 Cr)", "Small Cap (<₹5,000 Cr)"],
        index=0,
        help="Filter by market capitalization"
    )
    
    return {
        'expected_return': expected_return, 
        'risk_appetite': risk.lower(), 
        'holding_tenure': tenure,
        'price_range': (min_price, max_price),
        'market_cap': market_cap
    }


def render_signal_badge(signal: str, score: float):
    """Render signal badge with color."""
    colors = {'BUY': 'green', 'HOLD': 'orange', 'AVOID BUYING': 'red', 'SELL / EXIT': 'darkred'}
    color = colors.get(signal, 'gray')
    st.markdown(f"""
        <div style="background:{color}; color:white; padding:20px; border-radius:10px; text-align:center; margin:10px 0;">
            <h2 style="margin:0; color:white;">{signal}</h2>
            <p style="margin:5px 0 0 0; font-size:1.2em;">Score: {score:.0f}/100</p>
        </div>
    """, unsafe_allow_html=True)


def render_dimension_scores(scores: Dict[str, float]):
    """Render dimension scores."""
    st.subheader("Dimension Scores")
    for dim, score in scores.items():
        label = dim.replace('_', ' ').title()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(score / 100)
        with col2:
            st.markdown(f"**{label}**: {score:.0f}")


def render_why_not_buy(reasons: List[str]):
    """Render mandatory 'Why NOT to buy' section."""
    st.subheader("Why This May NOT Be Right For You")
    st.caption("Shown for ALL stocks, even BUY signals")
    with st.expander("View Important Considerations", expanded=True):
        for i, reason in enumerate(reasons, 1):
            icon = "📉" if "RETURN" in reason else "⚠️" if "RISK" in reason else "👔" if "GOVERNANCE" in reason or "PROMOTER" in reason else "💰" if "FINANCIAL" in reason or "CAPITAL" in reason else "📊" if "VALUATION" in reason else "🚩" if "RED FLAG" in reason else "ℹ️"
            st.markdown(f"{icon} **{i}.** {reason}")


def render_red_flags(flags: List[Dict]):
    """Render red flags."""
    if not flags:
        st.success("No red flags detected")
        return
    st.subheader("Red Flags Detected")
    for f in flags:
        sev = f.get('severity', 'medium')
        if sev == 'critical':
            st.error(f"🚨 CRITICAL: {f.get('description')}")
        elif sev == 'high':
            st.warning(f"⚠️ HIGH: {f.get('description')}")
        else:
            st.info(f"ℹ️ {sev.upper()}: {f.get('description')}")


def render_footer():
    """Render footer."""
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center; color:gray; font-size:0.8em;">
            <p>Decision support, not advice. <strong>Always do your own research.</strong></p>
            <p>Thanks: Yahoo Finance | NSE</p>
        </div>
    """, unsafe_allow_html=True)
