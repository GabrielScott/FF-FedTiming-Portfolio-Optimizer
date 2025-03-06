"""
Visualization Module

This module provides functions to create visualizations for portfolio analysis:
1. Factor exposure visualization
2. Regime-based return analysis
3. Portfolio performance dashboards
4. Interactive optimization tools
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualizations')

# Create visualization directory if it doesn't exist
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)


def load_data():
    """
    Load processed data for visualization
    
    Returns:
    --------
    tuple
        Tuple containing (stock_data, ff_factors, fed_decisions, fed_regimes)
    """
    logger.info("Loading processed data for visualization")
    
    try:
        # Import fed analysis functions for regime identification
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.fed_analysis import identify_fed_regimes
        
        # Find the most recent stock data file
        stock_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith('stock_prices_')]
        if not stock_files:
            stock_files = [f for f in os.listdir(DATA_DIR + '/raw') if f.startswith('stock_prices_')]
            if not stock_files:
                raise FileNotFoundError("No stock price data found")
        
        stock_file = sorted(stock_files)[-1]  # Most recent file
        stock_path = os.path.join(PROCESSED_DATA_DIR, stock_file)
        if not os.path.exists(stock_path):
            stock_path = os.path.join(DATA_DIR + '/raw', stock_file)
        
        # Load stock data
        stock_data = pd.read_csv(stock_path, index_col=0, parse_dates=True)
        
        # Load Fama-French factors
        ff_path = os.path.join(PROCESSED_DATA_DIR, "ff_factors_daily.csv")
        ff_factors = pd.read_csv(ff_path, index_col=0, parse_dates=True)
        
        # Load Fed decisions
        fed_path = os.path.join(PROCESSED_DATA_DIR, "fed_rate_decisions.csv")
        fed_decisions = pd.read_csv(fed_path, index_col=0, parse_dates=True)
        
        # Get Fed regimes
        fed_regimes = identify_fed_regimes(fed_decisions)
        
        logger.info("Data loaded successfully")
        return stock_data, ff_factors, fed_decisions, fed_regimes
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_factor_exposure_heatmap(factor_exposures, save_path=None):
    """
    Create a heatmap of factor exposures
    
    Parameters:
    -----------
    factor_exposures : dict
        Dictionary containing factor exposure results
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info("Creating factor exposure heatmap")
    
    # Get the latest factor exposures
    latest_betas = factor_exposures['betas'].iloc[-1].unstack(level=1)
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(latest_betas, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
    plt.title('Latest Factor Exposures by Asset')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Factor exposure heatmap saved to {save_path}")
    
    return plt.gcf()


def plot_regime_returns_distribution(returns_with_regime, save_path=None):
    """
    Plot distribution of returns by regime
    
    Parameters:
    -----------
    returns_with_regime : pandas.DataFrame
        DataFrame containing asset returns and regime data
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info("Plotting returns distribution by regime")
    
    # Get unique assets and regimes
    assets = returns_with_regime.columns.drop('regime')
    regimes = returns_with_regime['regime'].unique()
    
    # Create subplots, one row per asset
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)), sharex=True)
    
    # If only one asset, axes won't be an array
    if len(assets) == 1:
        axes = [axes]
    
    # Define colors for regimes
    regime_colors = {
        'Tightening': 'coral',
        'Easing': 'green',
        'Neutral': 'gray',
        'Unknown': 'blue'
    }
    
    # For each asset
    for i, asset in enumerate(assets):
        # Plot density by regime
        for regime in regimes:
            regime_data = returns_with_regime[returns_with_regime['regime'] == regime]
            if len(regime_data) > 10:  # Need sufficient data for kde
                sns.kdeplot(
                    regime_data[asset], 
                    ax=axes[i], 
                    label=regime,
                    color=regime_colors.get(regime, 'gray'),
                    fill=True, alpha=0.2
                )
        
        axes[i].set_title(f'Return Distribution for {asset} by Fed Regime')
        axes[i].set_ylabel('Density')
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].legend()
    
    axes[-1].set_xlabel('Return')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Regime returns distribution plot saved to {save_path}")
    
    return fig


def create_interactive_regime_dashboard(returns, fed_regimes, factor_exposures=None):
    """
    Create an interactive dashboard of regime-based portfolio analysis
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
    factor_exposures : dict, optional
        Dictionary containing factor exposure results
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the dashboard
    """
    logger.info("Creating interactive regime dashboard")
    
    # Create daily regime data
    daily_regime = pd.DataFrame(index=returns.index)
    for date, row in fed_regimes.iterrows():
        mask = (daily_regime.index >= date)
        daily_regime.loc[mask, 'regime'] = row['regime']
    
    # Forward fill any missing values
    daily_regime = daily_regime.fillna(method='ffill')
    daily_regime = daily_regime.fillna('Unknown')
    
    # Combine with returns
    returns_with_regime = pd.concat([returns, daily_regime], axis=1)
    
    # Calculate cumulative returns by regime
    cum_returns_by_regime = {}
    
    for regime in returns_with_regime['regime'].unique():
        regime_returns = returns_with_regime[returns_with_regime['regime'] == regime].drop('regime', axis=1)
        if len(regime_returns) > 0:
            cum_returns_by_regime[regime] = (1 + regime_returns).cumprod()
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns by Regime',
            'Return Distribution by Regime',
            'Average Monthly Returns by Regime',
            'Factor Exposures by Asset'
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "box"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.12
    )
    
    # Define regime colors
    regime_colors = {
        'Tightening': 'crimson',
        'Easing': 'forestgreen',
        'Neutral': 'darkgray',
        'Unknown': 'navy'
    }
    
    # Add cumulative return traces
    for regime, cum_rets in cum_returns_by_regime.items():
        for asset in cum_rets.columns:
            fig.add_trace(
                go.Scatter(
                    x=cum_rets.index,
                    y=cum_rets[asset],
                    name=f"{asset} ({regime})",
                    line=dict(color=regime_colors.get(regime, 'gray')),
                    visible='legendonly' if asset != returns.columns[0] else True,
                    hovertemplate=f"{asset} ({regime}): %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=1
            )
    
    # Calculate monthly returns for box plot
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_regime = daily_regime.resample('M').last()
    monthly_returns_with_regime = pd.concat([monthly_returns, monthly_regime], axis=1)
    
    # Prepare data for box plot
    for asset in monthly_returns.columns:
        data = []
        names = []
        
        for regime in monthly_returns_with_regime['regime'].unique():
            regime_data = monthly_returns_with_regime[monthly_returns_with_regime['regime'] == regime]
            if len(regime_data) > 0:
                data.append(regime_data[asset])
                names.append(regime)
        
        fig.add_trace(
            go.Box(
                y=data,
                name=asset,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=True,
                visible='legendonly' if asset != returns.columns[0] else True
            ),
            row=2, col=1
        )
    
    # Calculate average monthly returns by regime
    avg_monthly_returns = monthly_returns_with_regime.groupby('regime').mean()
    
    # Add average monthly returns bar chart
    for asset in monthly_returns.columns:
        fig.add_trace(
            go.Bar(
                x=avg_monthly_returns.index,
                y=avg_monthly_returns[asset],
                name=f"{asset} Avg Return",
                marker_color=[regime_colors.get(r, 'gray') for r in avg_monthly_returns.index],
                visible='legendonly' if asset != returns.columns[0] else True,
                text=[f"{val:.2%}" for val in avg_monthly_returns[asset]],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # Add factor exposure heatmap if provided
    if factor_exposures is not None:
        latest_betas = factor_exposures['betas'].iloc[-1].unstack(level=1)
        
        fig.add_trace(
            go.Heatmap(
                z=latest_betas.values,
                x=latest_betas.columns,
                y=latest_betas.index,
                colorscale='RdBu_r',
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in latest_betas.values],
                hovertemplate='Asset: %{y}<br>Factor: %{x}<br>Beta: %{z:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Analysis by Fed Regime',
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        boxmode='group'
    )
    
    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    
    fig.update_xaxes(title_text="Regime", row=2, col=1)
    fig.update_yaxes(title_text="Monthly Return", row=2, col=1)
    
    fig.update_xaxes(title_text="Factor", row=2, col=2)
    fig.update_yaxes(title_text="Asset", row=2, col=2)
    
    # Save as HTML file
    output_path = os.path.join(VISUALIZATION_DIR, 'regime_dashboard.html')
    fig.write_html(output_path)
    logger.info(f"Interactive regime dashboard saved to {output_path}")
    
    return fig


def create_portfolio_performance_dashboard(strategy_returns, weights, benchmark_returns=None, save_path=None):
    """
    Create interactive dashboard of portfolio performance
STRUCTURE:
Row 1, Col 1-2: Cumulative Returns
Row 2, Col 1: Drawdowns
Row 2, Col 2: Rolling 3-Month Returns
Row 3, Col 1: Rolling 6-Month Volatility
Row 3, Col 2: Return Distribution
Row 4, Col 1-2: Strategy Weights (pie charts

    
    
    Parameters:
    -----------
    strategy_returns : dict
        Dictionary mapping strategy names to returns Series
    weights : dict
        Dictionary mapping strategy names to weights DataFrames
    benchmark_returns : pandas.Series, optional
        Series containing benchmark returns
    save_path : str, optional
        Path to save the HTML file
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the dashboard
    """
    logger.info("Creating portfolio performance dashboard")
    
    # Create subplot figure
    fig = make_subplots(
        rows=4, cols=2,  # Increase to 4 rows to accommodate all chart types
        subplot_titles=(
            'Cumulative Returns',
            'Drawdowns',
            'Rolling 3-Month Returns',
            'Rolling 6-Month Volatility',
            'Return Distribution',
            'Strategy Weights'
        ),
        specs=[
            [{"colspan": 2}, None],      # Row 1: Cumulative returns (spans both columns)
            [{"type": "xy"}, {"type": "xy"}],  # Row 2: Drawdowns and volatility
            [{"type": "xy"}, {"type": "xy"}],  # Row 3: Rolling returns and distribution
            [{"type": "domain", "colspan": 2}, None]  # Row 4: Weights as pie chart
        ],
        vertical_spacing=0.08
    )
    
    # Add benchmark to strategies if provided
    all_returns = strategy_returns.copy()
    if benchmark_returns is not None:
        all_returns['Benchmark'] = benchmark_returns
    
    # Calculate metrics
    cumulative_returns = {}
    drawdowns = {}
    rolling_returns = {}
    rolling_volatility = {}
    
    for name, returns in all_returns.items():
        # Calculate cumulative returns
        cumulative_returns[name] = (1 + returns).cumprod()
        
        # Calculate drawdowns
        cum_ret = cumulative_returns[name]
        drawdowns[name] = cum_ret / cum_ret.cummax() - 1
        
        # Calculate rolling returns (3-month)
        rolling_returns[name] = returns.rolling(63).apply(lambda x: (1 + x).prod() - 1)
        
        # Calculate rolling volatility (6-month, annualized)
        rolling_volatility[name] = returns.rolling(126).std() * np.sqrt(252)
    
    # Add cumulative return traces on 1, 1
    for name, cum_ret in cumulative_returns.items():
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret,
                name=name,
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Add drawdown traces on 2, 1
    for name, dd in drawdowns.items():
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                name=f"{name} DD",
                hovertemplate=f"{name} Drawdown: %{{y:.2%}}<extra></extra>",
                visible='legendonly'
            ),
            row=2, col=1
        )
    
    # Add rolling return traces on 2, 2
    for name, roll_ret in rolling_returns.items():
        fig.add_trace(
            go.Scatter(
                x=roll_ret.index,
                y=roll_ret,
                name=f"{name} 3M",
                hovertemplate=f"{name} 3M Return: %{{y:.2%}}<extra></extra>",
                visible='legendonly'
            ),
            row=2, col=2
        )

    # Add rolling volatility on 3, 1
    for name, vol in rolling_volatility.items():
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol,
                name=f"{name} Vol",
                hovertemplate=f"{name} Volatility: %{{y:.2%}}<extra></extra>",
                visible=False
            ),
            row=3, col=1
        )

    
    # Add return distribution on 3, 2
    for name, returns in all_returns.items():
        fig.add_trace(
            go.Histogram(
                x=returns,
                name=f"{name} Dist",
                histnorm='probability density',
                opacity=0.6,
                nbinsx=30,
                visible='legendonly'
            ),
            row=3, col=2
        )
    

    
    # Add weights for strategies
    for name, weight_df in weights.items():
        if 'regime' in weight_df.columns:
            # Remove regime column for plotting
            weight_df = weight_df.drop('regime', axis=1)
        
        # Get latest weights
        latest_weights = weight_df.iloc[-1]
        
        # Add weights as stacked area chart
        for asset in weight_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=weight_df.index,
                    y=weight_df[asset],
                    name=f"{name} - {asset}",
                    stackgroup=name,
                    visible='legendonly'
                ),
                row=3, col=1
            )
        
        # Show latest weights as pie chart
        fig.add_trace(
            go.Pie(
                labels=latest_weights.index,
                values=latest_weights.values,
                name=f"{name} Weights",
                title=f"{name} Latest Weights",
                visible='legendonly'
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance Dashboard',
        height=1000,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="3-Month Return", row=2, col=2)

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=3, col=1)
    
    fig.update_xaxes(title_text="Daily Return", row=3, col=2)
    fig.update_yaxes(title_text="Probability Density", row=3, col=2)
    

    
    # Save as HTML file
    if save_path is None:
        save_path = os.path.join(VISUALIZATION_DIR, 'portfolio_dashboard.html')
    fig.write_html(save_path)
    logger.info(f"Portfolio performance dashboard saved to {save_path}")
    
    return fig


def create_performance_summary_table(performance_metrics, save_path=None):
    """
    Create a performance summary table with formatting
    
    Parameters:
    -----------
    performance_metrics : dict
        Dictionary mapping strategy names to performance metrics
    save_path : str, optional
        Path to save the HTML file
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the table
    """
    logger.info("Creating performance summary table")
    
    # Convert performance metrics to DataFrame
    metrics_df = pd.DataFrame(performance_metrics).T
    
    # Select key metrics for display
    key_metrics = [
        'total_return', 'cagr', 'volatility', 'sharpe_ratio', 
        'max_drawdown', 'sortino_ratio', 'calmar_ratio', 'win_rate'
    ]
    display_df = metrics_df[key_metrics].copy()
    
    # Format metrics
    formatted_df = display_df.copy()
    
    # Format percentages
    for col in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'win_rate']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('{:.2%}'.format)
    
    # Format ratios
    for col in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('{:.2f}'.format)
    
    # Rename columns for display
    column_names = {
        'total_return': 'Total Return',
        'cagr': 'CAGR',
        'volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'sortino_ratio': 'Sortino Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'win_rate': 'Win Rate'
    }
    formatted_df.columns = [column_names.get(col, col) for col in formatted_df.columns]
    
    # Create colored table
    colors = []
    for col in key_metrics:
        if col in ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
            col_colors = ['indianred' if val == display_df[col].max() else 
                         'lightcoral' if val > display_df[col].mean() else 'white' 
                         for val in display_df[col]]
        else:  # For risk metrics lower is better
            col_colors = ['indianred' if val == display_df[col].min() else 
                         'lightcoral' if val < display_df[col].mean() else 'white' 
                         for val in display_df[col]]
        colors.append(col_colors)
    
    # Transpose for plotly format
    colors = list(map(list, zip(*colors)))
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Strategy'] + [column_names.get(col, col) for col in key_metrics],
            fill_color='royalblue',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[formatted_df.index] + [formatted_df[col] for col in formatted_df.columns],
            fill_color=[['lightgrey'] * len(formatted_df)] + colors,
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title='Strategy Performance Comparison',
        height=400,
        width=1000
    )
    
    # Save as HTML file
    if save_path is None:
        save_path = os.path.join(VISUALIZATION_DIR, 'performance_table.html')
    fig.write_html(save_path)
    logger