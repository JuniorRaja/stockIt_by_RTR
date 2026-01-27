"""Chart components for Indian Equity Intelligence."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Common chart configuration with zoom reset and other useful tools
CHART_CONFIG = {
    'displayModeBar': True,
    'modeBarButtonsToAdd': ['resetScale2d', 'resetViews'],
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'displaylogo': False,
    'scrollZoom': True,
}


def create_price_chart(price_history: pd.DataFrame, title: str = "Price History", show_volume: bool = True) -> go.Figure:
    """Create interactive price chart with full date range."""
    # Ensure we use all available data
    df = price_history.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    if show_volume and 'volume' in df.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'],
                                      high=df['high'], low=df['low'],
                                      close=df['close'], name='Price'), row=1, col=1)
        colors = ['red' if r['close'] < r['open'] else 'green' for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'],
                             marker_color=colors, name='Volume', opacity=0.5), row=2, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'],
                                      high=df['high'], low=df['low'],
                                      close=df['close'], name='Price'))
        fig.update_yaxes(title_text="Price (₹)")
    
    # Add date range info to title
    if 'date' in df.columns and len(df) > 0:
        start_year = df['date'].min().year
        end_year = df['date'].max().year
        title = f"{title} ({start_year} - {end_year})"
    
    fig.update_layout(
        title=title, 
        xaxis_rangeslider_visible=False, 
        template="plotly_dark", 
        height=500,
        # Add range selector buttons for quick navigation
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]),
                bgcolor="rgba(50, 50, 50, 0.8)",
                activecolor="rgba(100, 100, 100, 0.8)",
            ),
            type="date"
        )
    )
    return fig


def get_chart_config():
    """Return chart config for Streamlit plotly_chart calls."""
    return CHART_CONFIG


def create_score_radar(scores: Dict[str, float]) -> go.Figure:
    """Create radar chart for dimension scores."""
    categories = [k.replace('_', ' ').title() for k in scores.keys()]
    values = list(scores.values())
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Score',
                                   fillcolor='rgba(0, 150, 255, 0.3)', line=dict(color='rgb(0, 150, 255)')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=False, title="Analysis Scores", template="plotly_dark", height=400)
    return fig


def create_drawdown_chart(price_history: pd.DataFrame) -> go.Figure:
    """Create drawdown chart with full history."""
    df = price_history.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    prices = df.set_index('date')['close']
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max * 100
    
    # Get date range for title
    start_year = prices.index.min().year
    end_year = prices.index.max().year
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='Drawdown',
                             fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red')))
    
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    fig.add_trace(go.Scatter(x=[max_dd_idx], y=[max_dd], mode='markers+text', name='Max DD',
                             marker=dict(size=12, color='darkred'), text=[f'{max_dd:.1f}%'], textposition='bottom center'))
    
    fig.update_layout(
        title=f"Drawdown History ({start_year} - {end_year})", 
        xaxis_title="Date", 
        yaxis_title="Drawdown %",
        template="plotly_dark", 
        height=350, 
        yaxis=dict(range=[min(-60, max_dd * 1.1), 5]),
        # Add range selector
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]),
                bgcolor="rgba(50, 50, 50, 0.8)",
            ),
            type="date"
        )
    )
    return fig


def create_governance_chart(shareholding: pd.DataFrame) -> go.Figure:
    """Create shareholding pattern chart."""
    fig = go.Figure()
    if 'promoter_holding' in shareholding.columns:
        fig.add_trace(go.Scatter(x=shareholding['date'], y=shareholding['promoter_holding'],
                                 mode='lines+markers', name='Promoter', line=dict(color='green', width=2),
                                 fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'))
    if 'fii_holding' in shareholding.columns:
        fig.add_trace(go.Scatter(x=shareholding['date'], y=shareholding['fii_holding'],
                                 mode='lines+markers', name='FII', line=dict(color='blue')))
    if 'promoter_pledge' in shareholding.columns:
        fig.add_trace(go.Scatter(x=shareholding['date'], y=shareholding['promoter_pledge'],
                                 mode='lines+markers', name='Pledge', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title="Shareholding Pattern", xaxis_title="Date", yaxis_title="Holding %",
                      template="plotly_dark", height=400, yaxis=dict(range=[0, 100]))
    return fig


def create_scenario_chart(results: Dict) -> go.Figure:
    """Create scenario comparison chart."""
    scenarios, impacts, resilience = [], [], []
    for name, r in results.items():
        scenarios.append(r.scenario.name)
        impacts.append(r.impact_summary['return_impact'])
        resilience.append({'robust': 4, 'resilient': 3, 'moderate': 2, 'fragile': 1}.get(r.resilience_rating, 2))
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Return Impact", "Resilience"))
    colors = ['green' if x >= 0 else 'red' for x in impacts]
    fig.add_trace(go.Bar(x=scenarios, y=impacts, marker_color=colors, name='Impact'), row=1, col=1)
    res_colors = ['green' if x >= 3 else 'orange' if x >= 2 else 'red' for x in resilience]
    fig.add_trace(go.Bar(x=scenarios, y=resilience, marker_color=res_colors, name='Resilience'), row=1, col=2)
    
    fig.update_layout(title="Scenario Analysis", template="plotly_dark", height=400, showlegend=False)
    return fig
