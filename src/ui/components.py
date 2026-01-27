"""Streamlit UI Components for Indian Equity Intelligence."""

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
    st.subheader("Your Investment Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        expected_return = st.slider("Expected CAGR %", 5, 30, 15, help="Annual return expectation")
    with col2:
        risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"], index=1)
    with col3:
        tenure = st.slider("Holding Period (Years)", 1, 10, 5)
    return {'expected_return': expected_return, 'risk_appetite': risk.lower(), 'holding_tenure': tenure}


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
            <p><strong>Indian Equity Intelligence</strong> - Local-first Stock Analysis</p>
            <p>Decision support, not advice. Always do your own research.</p>
            <p>Data: Yahoo Finance, NSE | No external data transmission</p>
        </div>
    """, unsafe_allow_html=True)
