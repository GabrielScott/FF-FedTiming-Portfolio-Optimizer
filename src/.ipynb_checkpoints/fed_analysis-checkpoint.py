"""
Fed Analysis Module

This module analyzes the relationship between Federal Reserve interest rate
decisions and asset returns. It includes functions to:
1. Calculate returns around Fed decision dates
2. Analyze market regimes based on Fed policy
3. Create trading signals based on Fed decisions
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


def load_data():
    """
    Load processed data for analysis
    
    Returns:
    --------
    tuple
        Tuple containing (stock_data, ff_factors, fed_decisions)
    """
    logger.info("Loading processed data for Fed analysis")
    
    try:
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
        
        logger.info("Data loaded successfully")
        return stock_data, ff_factors, fed_decisions
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def calculate_returns(prices, periods=[1, 5, 10, 20]):
    """
    Calculate returns over different time periods
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame containing price data
    periods : list, optional
        List of periods to calculate returns for
        
    Returns:
    --------
    dict
        Dictionary mapping period to DataFrame of returns
    """
    logger.info(f"Calculating returns for periods: {periods}")
    
    returns = {}
    
    # Calculate simple returns for each period
    for period in periods:
        period_returns = prices.pct_change(period)
        returns[period] = period_returns
    
    # Also add daily returns
    returns[1] = prices.pct_change()
    
    return returns


def calculate_fed_event_returns(stock_data, fed_decisions, windows=[-10, -5, -1, 0, 1, 5, 10]):
    """
    Calculate returns around Fed decision dates
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        DataFrame containing stock price data
    fed_decisions : pandas.DataFrame
        DataFrame containing Fed decision dates
    windows : list, optional
        List of days relative to Fed decisions to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing returns for each window around Fed decisions
    """
    logger.info(f"Calculating returns around Fed events for windows: {windows}")
    
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    
    # Initialize dictionary to store results
    event_returns = {}
    
    # Get Fed decision dates that are in the range of our stock data
    fed_dates = fed_decisions.index
    valid_fed_dates = [date for date in fed_dates if date in daily_returns.index]
    
    if not valid_fed_dates:
        logger.warning("No Fed decision dates found in stock data range")
        return pd.DataFrame()
    
    # For each window pair, calculate cumulative returns
    for i in range(len(windows) - 1):
        start_offset = windows[i]
        end_offset = windows[i + 1]
        window_name = f"{start_offset}to{end_offset}"
        
        window_returns = []
        
        # For each Fed decision date
        for fed_date in valid_fed_dates:
            try:
                # Find the corresponding index positions
                fed_idx = daily_returns.index.get_loc(fed_date)
                start_idx = max(0, fed_idx + start_offset)
                end_idx = min(len(daily_returns) - 1, fed_idx + end_offset)
                
                # Skip if we don't have enough data
                if start_idx >= end_idx:
                    continue
                
                # Calculate cumulative return for this window
                cum_return = (daily_returns.iloc[start_idx:end_idx + 1] + 1).prod() - 1
                cum_return = cum_return.to_frame().T
                
                # Add metadata
                cum_return['fed_date'] = fed_date
                cum_return['decision'] = fed_decisions.loc[fed_date, 'decision']
                cum_return['rate_change'] = fed_decisions.loc[fed_date, 'change']
                
                window_returns.append(cum_return)
            except:
                logger.warning(f"Error processing Fed date {fed_date} for window {window_name}")
                continue
        
        if window_returns:
            event_returns[window_name] = pd.concat(window_returns)
    
    # Return results as a dict mapping window name to DataFrame
    return event_returns


def identify_fed_regimes(fed_decisions, window_size=3):
    """
    Identify monetary policy regimes based on Fed decisions
    
    Parameters:
    -----------
    fed_decisions : pandas.DataFrame
        DataFrame containing Fed decision dates
    window_size : int, optional
        Number of decisions to look back to determine regime
        
    Returns:
    --------
    pandas.DataFrame
        Fed decisions with regime labels added
    """
    logger.info(f"Identifying Fed regimes with window size {window_size}")
    
    # Create a copy to avoid modifying the original
    fed_with_regimes = fed_decisions.copy()
    
    # Calculate rolling sum of rate changes
    fed_with_regimes['rolling_change'] = fed_with_regimes['change'].rolling(window=window_size, min_periods=1).sum()
    
    # Determine regime based on rolling sum
    conditions = [
        fed_with_regimes['rolling_change'] > 0,
        fed_with_regimes['rolling_change'] < 0,
        fed_with_regimes['rolling_change'] == 0
    ]
    choices = ['Tightening', 'Easing', 'Neutral']
    
    fed_with_regimes['regime'] = np.select(conditions, choices, default='Unknown')
    
    # Add a flag for regime changes
    fed_with_regimes['regime_change'] = fed_with_regimes['regime'].shift(1) != fed_with_regimes['regime']
    fed_with_regimes.loc[fed_with_regimes.index[0], 'regime_change'] = True
    
    return fed_with_regimes


def analyze_returns_by_regime(stock_data, fed_with_regimes):
    """
    Analyze asset returns based on Fed policy regimes
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        DataFrame containing stock price data
    fed_with_regimes : pandas.DataFrame
        DataFrame containing Fed decisions with regime labels
        
    Returns:
    --------
    pandas.DataFrame
        Summary statistics of returns by regime
    """
    logger.info("Analyzing returns by Fed regime")
    
    # Calculate monthly returns
    monthly_returns = stock_data.resample('M').last().pct_change().dropna()
    
    # Create a daily regime series
    daily_regime = pd.DataFrame(index=stock_data.index)
    
    # Assign regimes to each day based on the most recent Fed decision
    current_regime = 'Unknown'
    
    for date, row in fed_with_regimes.iterrows():
        mask = (daily_regime.index >= date)
        daily_regime.loc[mask, 'regime'] = row['regime']
        
    # Merge regimes with monthly returns
    monthly_regime = daily_regime.resample('M').last()
    returns_with_regime = pd.concat([monthly_returns, monthly_regime], axis=1).dropna()
    
    # Calculate statistics by regime
    regime_stats = {}
    
    for regime in returns_with_regime['regime'].unique():
        regime_data = returns_with_regime[returns_with_regime['regime'] == regime].drop('regime', axis=1)
        
        if len(regime_data) > 0:
            stats = {
                'mean': regime_data.mean(),
                'median': regime_data.median(),
                'std': regime_data.std(),
                'sharpe': regime_data.mean() / regime_data.std() * np.sqrt(12),  # Annualized Sharpe
                'win_rate': (regime_data > 0).mean(),
                'count': len(regime_data)
            }
            
            regime_stats[regime] = pd.DataFrame(stats).T
    
    # Combine all regime statistics
    all_stats = pd.concat(regime_stats, axis=0)
    all_stats.index.names = ['regime', 'stat']
    
    return all_stats


def create_fed_timing_signal(fed_with_regimes, signal_type='regime_change'):
    """
    Create trading signals based on Fed decisions
    
    Parameters:
    -----------
    fed_with_regimes : pandas.DataFrame
        DataFrame containing Fed decisions with regime labels
    signal_type : str, optional
        Type of signal to create
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with trading signals
    """
    logger.info(f"Creating Fed timing signals of type: {signal_type}")
    
    signals = fed_with_regimes.copy()
    
    if signal_type == 'regime_change':
        # Signal is 1 at regime changes, 0 otherwise
        signals['signal'] = signals['regime_change'].astype(int)
        
    elif signal_type == 'decision':
        # Signal based on decision type
        signals['signal'] = 0
        signals.loc[signals['decision'] == 'Cut', 'signal'] = 1
        signals.loc[signals['decision'] == 'Hike', 'signal'] = -1
        
    elif signal_type == 'surprise':
        # This would use a model of Fed decision surprises
        # For simplicity, we'll use a random signal here
        signals['signal'] = np.random.choice([-1, 0, 1], size=len(signals))
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    # Expand signal to daily frequency
    daily_signal = pd.DataFrame(index=pd.date_range(
        start=signals.index.min(),
        end=signals.index.max(),
        freq='B'
    ))
    
    # Forward fill the signal
    for date, row in signals.iterrows():
        daily_signal.loc[date:, 'signal'] = row['signal']
        daily_signal.loc[date:, 'regime'] = row['regime']
    
    # Forward fill any missing values
    daily_signal = daily_signal.fillna(method='ffill')
    
    return daily_signal


def plot_fed_regimes(fed_with_regimes, asset_prices=None):
    """
    Plot Fed regimes and optionally overlay with asset prices
    
    Parameters:
    -----------
    fed_with_regimes : pandas.DataFrame
        DataFrame containing Fed decisions with regime labels
    asset_prices : pandas.DataFrame, optional
        DataFrame containing asset prices to overlay
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info("Plotting Fed regimes")
    
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot Fed Funds Rate
    axes[0].plot(fed_with_regimes.index, fed_with_regimes['rate'], marker='o', linestyle='-', linewidth=2)
    axes[0].set_title('Federal Funds Target Rate')
    axes[0].set_ylabel('Rate (%)')
    axes[0].grid(True)
    
    # Color the background based on regime
    regime_colors = {
        'Tightening': 'lightcoral',
        'Easing': 'lightgreen',
        'Neutral': 'lightyellow',
        'Unknown': 'lightgray'
    }
    
    # Find regime change dates
    regime_changes = fed_with_regimes[fed_with_regimes['regime_change']].index.tolist()
    if regime_changes:
        regime_changes = [fed_with_regimes.index[0]] + regime_changes + [fed_with_regimes.index[-1]]
        
        # Color each regime period
        for i in range(len(regime_changes) - 1):
            start_date = regime_changes[i]
            end_date = regime_changes[i + 1]
            regime = fed_with_regimes.loc[start_date, 'regime']
            color = regime_colors.get(regime, 'lightgray')
            
            # Add colored background
            axes[0].axvspan(start_date, end_date, alpha=0.3, color=color)
            axes[1].axvspan(start_date, end_date, alpha=0.3, color=color)
    
    # Plot asset prices if provided
    if asset_prices is not None:
        for column in asset_prices.columns:
            # Normalize to 100 at the start
            normalized = asset_prices[column] / asset_prices[column].iloc[0] * 100
            axes[1].plot(normalized.index, normalized, label=column)
        
        axes[1].set_title('Asset Prices (Normalized to 100)')
        axes[1].set_ylabel('Index Value')
        axes[1].legend()
    else:
        axes[1].set_visible(False)
    
    axes[1].set_xlabel('Date')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    return fig


def analyze_fed_impact_on_factors(ff_factors, fed_with_regimes):
    """
    Analyze how Fed policy regimes affect Fama-French factors
    
    Parameters:
    -----------
    ff_factors : pandas.DataFrame
        DataFrame containing Fama-French factors
    fed_with_regimes : pandas.DataFrame
        DataFrame containing Fed decisions with regime labels
        
    Returns:
    --------
    pandas.DataFrame
        Factor performance statistics by regime
    """
    logger.info("Analyzing Fed impact on Fama-French factors")
    
    # Create a daily regime series
    daily_regime = pd.DataFrame(index=ff_factors.index)
    
    # Assign regimes to each day based on the most recent Fed decision
    for date, row in fed_with_regimes.iterrows():
        mask = (daily_regime.index >= date)
        daily_regime.loc[mask, 'regime'] = row['regime']
    
    # Forward fill any missing values
    daily_regime = daily_regime.fillna(method='ffill')
    daily_regime = daily_regime.fillna('Unknown')  # For dates before first Fed decision
    
    # Combine FF factors with regimes
    factors_with_regime = pd.concat([ff_factors, daily_regime], axis=1).dropna()
    
    # Calculate statistics by regime
    regime_stats = {}
    
    for regime in factors_with_regime['regime'].unique():
        regime_data = factors_with_regime[factors_with_regime['regime'] == regime].drop('regime', axis=1)
        
        if len(regime_data) > 0:
            # Calculate annualization factor based on the data frequency
            if len(regime_data) > 252:  # Assuming daily data
                annualization_factor = np.sqrt(252)
            elif len(regime_data) > 52:  # Assuming weekly data
                annualization_factor = np.sqrt(52)
            else:  # Assuming monthly data
                annualization_factor = np.sqrt(12)
            
            stats = {
                'mean': regime_data.mean() * annualization_factor,  # Annualized mean
                'median': regime_data.median() * annualization_factor,  # Annualized median
                'std': regime_data.std() * np.sqrt(annualization_factor),  # Annualized std
                'sharpe': regime_data.mean() / regime_data.std() * annualization_factor,  # Annualized Sharpe
                'win_rate': (regime_data > 0).mean(),
                'count': len(regime_data),
                'days': len(regime_data)
            }
            
            regime_stats[regime] = pd.DataFrame(stats).T
    
    # Combine all regime statistics
    all_stats = pd.concat(regime_stats, axis=0)
    all_stats.index.names = ['regime', 'stat']
    
    return all_stats


if __name__ == "__main__":
    # Example usage
    stock_data, ff_factors, fed_decisions = load_data()
    
    # Identify Fed regimes
    fed_with_regimes = identify_fed_regimes(fed_decisions)
    
    # Calculate returns around Fed events
    event_returns = calculate_fed_event_returns(stock_data, fed_decisions)
    
    # Analyze returns by regime
    regime_stats = analyze_returns_by_regime(stock_data, fed_with_regimes)
    
    # Create Fed timing signal
    timing_signal = create_fed_timing_signal(fed_with_regimes)
    
    # Analyze Fed impact on factors
    factor_regime_stats = analyze_fed_impact_on_factors(ff_factors, fed_with_regimes)
    
    # Plot Fed regimes
    fig = plot_fed_regimes(fed_with_regimes, stock_data)
    plt.savefig(os.path.join(DATA_DIR, 'fed_regimes.png'))
    plt.close(fig)
    
    print("Fed analysis completed successfully")