"""
Factor Modeling Module

This module provides functions to:
1. Calculate asset exposures to Fama-French factors
2. Estimate expected returns based on factor exposures
3. Implement time-varying factor models conditioned on Fed policy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


def load_data():
    """
    Load processed data for factor modeling
    
    Returns:
    --------
    tuple
        Tuple containing (stock_data, ff_factors, fed_decisions)
    """
    logger.info("Loading processed data for factor modeling")
    
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


def calculate_asset_returns(prices, frequency='daily'):
    """
    Calculate asset returns at specified frequency
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame containing price data
    frequency : str, optional
        Frequency of returns, default is 'daily'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing returns at specified frequency
    """
    logger.info(f"Calculating {frequency} returns")
    
    # Calculate returns
    if frequency == 'daily':
        returns = prices.pct_change().dropna()
    elif frequency == 'weekly':
        returns = prices.resample('W').last().pct_change().dropna()
    elif frequency == 'monthly':
        returns = prices.resample('M').last().pct_change().dropna()
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    return returns


def estimate_factor_exposures(returns, factors, window=252):
    """
    Estimate rolling factor exposures (betas) for assets
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    factors : pandas.DataFrame
        DataFrame containing factor returns
    window : int, optional
        Rolling window size for estimation
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames for factor exposures and model statistics
    """
    logger.info(f"Estimating factor exposures with {window}-day window")
    
    # Align dates between returns and factors
    aligned_data = pd.concat([returns, factors], axis=1).dropna()
    
    # Extract returns and factors
    asset_returns = aligned_data[returns.columns]
    factor_returns = aligned_data[factors.columns]
    
    # Store results
    factor_betas = {}
    factor_alpha = {}
    factor_r2 = {}
    
    # For each asset, estimate factor exposures
    for asset in asset_returns.columns:
        logger.info(f"Estimating factor exposures for {asset}")
        
        # Set up the regression model (y ~ X)
        y = asset_returns[asset]
        X = factor_returns[['mkt_excess', 'smb', 'hml']]
        X = add_constant(X)
        
        # Perform rolling regression
        rolling_model = RollingOLS(y, X, window=window)
        rolling_results = rolling_model.fit()
        
        # Extract results
        params = rolling_results.params
        
        # Store factor exposures (betas)
        factor_betas[asset] = params.drop('const', axis=1)
        
        # Store alpha
        factor_alpha[asset] = params['const']
        
        # Calculate R-squared (this is a simplified approach)
        pred = (X.values @ params.values.T)  # Matrix multiplication
        resid = y - pd.Series(pred, index=y.index)
        r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        factor_r2[asset] = pd.Series(r2, index=params.index)
    
    # Combine results into DataFrames
    beta_df = pd.concat(factor_betas, axis=1)
    beta_df.columns.names = ['asset', 'factor']
    
    alpha_df = pd.DataFrame(factor_alpha)
    r2_df = pd.DataFrame(factor_r2)
    
    return {
        'betas': beta_df,
        'alpha': alpha_df,
        'r2': r2_df
    }


def calculate_time_varying_exposures(returns, factors, fed_regimes, window=126):
    """
    Calculate time-varying factor exposures conditional on Fed regime
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    factors : pandas.DataFrame
        DataFrame containing factor returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
    window : int, optional
        Window size for rolling estimation
        
    Returns:
    --------
    dict
        Dictionary containing conditional exposures by regime
    """
    logger.info(f"Calculating time-varying factor exposures with {window}-day window")
    
    # Align dates across all inputs
    aligned_data = pd.concat([returns, factors, fed_regimes[['regime']]], axis=1).dropna()
    
    # Group by regime
    regimes = aligned_data['regime'].unique()
    
    # Store results
    regime_exposures = {}
    
    for regime in regimes:
        logger.info(f"Estimating exposures for regime: {regime}")
        
        # Filter data for this regime
        regime_data = aligned_data[aligned_data['regime'] == regime]
        
        if len(regime_data) < window:
            logger.warning(f"Insufficient data for regime {regime}, skipping")
            continue
        
        # Extract returns and factors
        regime_returns = regime_data[returns.columns]
        regime_factors = regime_data[factors.columns[:-1]]  # Exclude the 'regime' column
        
        # Estimate exposures for this regime
        regime_exposures[regime] = {}
        
        for asset in regime_returns.columns:
            # Set up the regression model
            y = regime_returns[asset]
            X = regime_factors[['mkt_excess', 'smb', 'hml']]
            X = add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X)
            results = model.fit()
            
            # Store the results
            regime_exposures[regime][asset] = {
                'params': results.params,
                'std_errors': results.bse,
                't_values': results.tvalues,
                'p_values': results.pvalues,
                'r2': results.rsquared,
                'adj_r2': results.rsquared_adj
            }
    
    return regime_exposures


def calculate_factor_expected_returns(factor_exposures, factor_premiums):
    """
    Calculate expected returns based on factor exposures and expected factor premiums
    
    Parameters:
    -----------
    factor_exposures : dict
        Dictionary containing factor exposures by asset
    factor_premiums : pandas.Series
        Series containing expected factor premiums
        
    Returns:
    --------
    pandas.Series
        Series containing expected returns for each asset
    """
    logger.info("Calculating expected returns from factor exposures")
    
    expected_returns = {}
    
    for asset, exposures in factor_exposures.items():
        # Calculate expected return: sum(beta_i * premium_i) + alpha
        expected_return = 0
        
        for factor, beta in exposures.items():
            if factor == 'const':
                expected_return += beta  # Add alpha
            else:
                expected_return += beta * factor_premiums.get(factor, 0)
        
        expected_returns[asset] = expected_return
    
    return pd.Series(expected_returns)


def estimate_factor_premiums_by_regime(factors, fed_regimes):
    """
    Estimate factor premiums conditional on Fed regime
    
    Parameters:
    -----------
    factors : pandas.DataFrame
        DataFrame containing factor returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing factor premiums by regime
    """
    logger.info("Estimating factor premiums by Fed regime")
    
    # Align dates
    aligned_data = pd.concat([factors, fed_regimes[['regime']]], axis=1).dropna()
    
    # Calculate annualization factor based on frequency
    obs_per_year = min(252, len(aligned_data))  # Default to 252 for daily data
    annualization_factor = np.sqrt(obs_per_year)
    
    # Group by regime and calculate statistics
    regime_premiums = {}
    
    for regime, group in aligned_data.groupby('regime'):
        # Extract factor returns for this regime
        regime_factors = group.drop('regime', axis=1)
        
        # Calculate factor premiums (annualized)
        mean_returns = regime_factors.mean() * obs_per_year
        std_returns = regime_factors.std() * annualization_factor
        sharpe_ratios = mean_returns / std_returns
        t_stats = mean_returns / (std_returns / np.sqrt(len(regime_factors)))
        
        regime_premiums[regime] = pd.DataFrame({
            'mean': mean_returns,
            'std': std_returns,
            'sharpe': sharpe_ratios,
            't_stat': t_stats,
            'count': len(regime_factors)
        })
    
    # Combine results
    factor_premiums_by_regime = pd.concat(regime_premiums, axis=0)
    factor_premiums_by_regime.index.names = ['regime', 'factor']
    
    return factor_premiums_by_regime


def plot_rolling_betas(factor_exposures, asset=None):
    """
    Plot rolling factor betas for a specific asset or all assets
    
    Parameters:
    -----------
    factor_exposures : dict
        Dictionary containing factor exposure results
    asset : str, optional
        Specific asset to plot, if None will plot for all assets
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info(f"Plotting rolling betas for {asset if asset else 'all assets'}")
    
    betas = factor_exposures['betas']
    
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    if asset:
        # Plot for a specific asset
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for factor in ['mkt_excess', 'smb', 'hml']:
            ax.plot(betas.index, betas[(asset, factor)], label=f'{factor.upper()}')
        
        ax.set_title(f'Rolling Factor Betas for {asset}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Beta')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.legend()
        
    else:
        # Plot for all assets (one subplot per asset)
        assets = betas.columns.get_level_values(0).unique()
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)), sharex=True)
        
        for i, asset in enumerate(assets):
            ax = axes[i] if len(assets) > 1 else axes
            
            for factor in ['mkt_excess', 'smb', 'hml']:
                ax.plot(betas.index, betas[(asset, factor)], label=f'{factor.upper()}')
            
            ax.set_title(f'Rolling Factor Betas for {asset}')
            ax.set_ylabel('Beta')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.legend()
        
        axes[-1].set_xlabel('Date') if len(assets) > 1 else axes.set_xlabel('Date')
    
    plt.tight_layout()
    
    return fig


def plot_factor_premiums_by_regime(factor_premiums_by_regime):
    """
    Plot factor premiums by Fed regime
    
    Parameters:
    -----------
    factor_premiums_by_regime : pandas.DataFrame
        DataFrame containing factor premiums by regime
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info("Plotting factor premiums by regime")
    
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Extract the mean returns for plotting
    mean_returns = factor_premiums_by_regime.xs('mean', level=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    regimes = mean_returns.index.tolist()
    factors = mean_returns.columns.tolist()
    x = np.arange(len(regimes))
    width = 0.2
    
    # Plot bars for each factor
    for i, factor in enumerate(factors):
        offset = (i - len(factors) / 2 + 0.5) * width
        ax.bar(x + offset, mean_returns[factor], width, label=factor.upper())
    
    # Customize plot
    ax.set_title('Factor Premiums by Fed Regime (Annualized)', fontsize=16)
    ax.set_xlabel('Fed Regime', fontsize=14)
    ax.set_ylabel('Annualized Return', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, factor in enumerate(factors):
        offset = (i - len(factors) / 2 + 0.5) * width
        for j, regime in enumerate(regimes):
            value = mean_returns.loc[regime, factor]
            ax.annotate(f'{value:.2%}',
                        xy=(j + offset, value),
                        xytext=(0, 3 if value >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if value >= 0 else 'top',
                        fontsize=8)
    
    plt.tight_layout()
    
    return fig


def implement_time_varying_factor_model(returns, factors, fed_regimes, window=252, prediction_window=63):
    """
    Implement a time-varying factor model with Fed regime conditioning
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    factors : pandas.DataFrame
        DataFrame containing factor returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
    window : int, optional
        Window size for rolling estimation
    prediction_window : int, optional
        Window size for prediction evaluation
        
    Returns:
    --------
    dict
        Dictionary containing model performance results
    """
    logger.info(f"Implementing time-varying factor model with {window}-day window")
    
    # Align dates
    aligned_data = pd.concat([returns, factors, fed_regimes[['regime']]], axis=1).dropna()
    
    # Create time series splits for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Store results
    model_results = {
        'unconditional': {'mae': [], 'rmse': [], 'ic': []},
        'regime_conditional': {'mae': [], 'rmse': [], 'ic': []}
    }
    
    # Track predictions
    all_predictions = {
        'unconditional': [],
        'regime_conditional': [],
        'actual': []
    }
    
    # For each time split
    for train_idx, test_idx in tscv.split(aligned_data):
        train_data = aligned_data.iloc[train_idx]
        test_data = aligned_data.iloc[test_idx]
        
        # Check if we have enough data for both regimes
        regimes_in_train = train_data['regime'].unique()
        if len(regimes_in_train) < 2:
            logger.warning(f"Training data contains only {regimes_in_train} regime(s), skipping this split")
            continue
        
        # Train unconditional model
        unconditional_predictions = {}
        
        for asset in returns.columns:
            # Set up the regression model
            y_train = train_data[asset]
            X_train = train_data[['mkt_excess', 'smb', 'hml']]
            X_train = add_constant(X_train)
            
            # Fit the model
            model = sm.OLS(y_train, X_train)
            results = model.fit()
            
            # Make predictions on test set
            X_test = test_data[['mkt_excess', 'smb', 'hml']]
            X_test = add_constant(X_test)
            pred = results.predict(X_test)
            
            unconditional_predictions[asset] = pred
        
        # Combine unconditional predictions
        unconditional_df = pd.DataFrame(unconditional_predictions, index=test_data.index)
        
        # Train regime-conditional models
        regime_conditional_predictions = {}
        
        for asset in returns.columns:
            predictions = []
            
            # For each regime in the test set
            for regime, regime_test in test_data.groupby('regime'):
                # Filter training data for this regime
                regime_train = train_data[train_data['regime'] == regime]
                
                if len(regime_train) < 30:  # Minimum observations needed for regression
                    logger.warning(f"Insufficient training data for regime {regime}, using unconditional model")
                    X_test = regime_test[['mkt_excess', 'smb', 'hml']]
                    X_test = add_constant(X_test)
                    pred = results.predict(X_test)  # Use unconditional model
                else:
                    # Train regime-specific model
                    y_train = regime_train[asset]
                    X_train = regime_train[['mkt_excess', 'smb', 'hml']]
                    X_train = add_constant(X_train)
                    
                    regime_model = sm.OLS(y_train, X_train)
                    regime_results = regime_model.fit()
                    
                    # Make predictions
                    X_test = regime_test[['mkt_excess', 'smb', 'hml']]
                    X_test = add_constant(X_test)
                    pred = regime_results.predict(X_test)
                
                predictions.append(pd.Series(pred, index=regime_test.index))
            
            # Combine predictions for all regimes
            if predictions:
                regime_conditional_predictions[asset] = pd.concat(predictions).sort_index()
        
        # Combine regime-conditional predictions
        regime_conditional_df = pd.DataFrame(regime_conditional_predictions, index=test_data.index)
        
        # Evaluate predictions
        actual_returns = test_data[returns.columns]
        
        # Calculate MAE
        unconditional_mae = (actual_returns - unconditional_df).abs().mean()
        regime_conditional_mae = (actual_returns - regime_conditional_df).abs().mean()
        
        # Calculate RMSE
        unconditional_rmse = np.sqrt(((actual_returns - unconditional_df) ** 2).mean())
        regime_conditional_rmse = np.sqrt(((actual_returns - regime_conditional_df) ** 2).mean())
        
        # Calculate Information Coefficient (correlation between predicted and actual returns)
        unconditional_ic = {}
        regime_conditional_ic = {}
        
        for asset in returns.columns:
            unconditional_corr = actual_returns[asset].corr(unconditional_df[asset])
            regime_conditional_corr = actual_returns[asset].corr(regime_conditional_df[asset])
            
            unconditional_ic[asset] = unconditional_corr
            regime_conditional_ic[asset] = regime_conditional_corr
        
        # Store results
        model_results['unconditional']['mae'].append(unconditional_mae)
        model_results['unconditional']['rmse'].append(unconditional_rmse)
        model_results['unconditional']['ic'].append(pd.Series(unconditional_ic))
        
        model_results['regime_conditional']['mae'].append(regime_conditional_mae)
        model_results['regime_conditional']['rmse'].append(regime_conditional_rmse)
        model_results['regime_conditional']['ic'].append(pd.Series(regime_conditional_ic))
        
        # Store predictions for this split
        all_predictions['unconditional'].append(unconditional_df)
        all_predictions['regime_conditional'].append(regime_conditional_df)
        all_predictions['actual'].append(actual_returns)
    
    # Combine results across splits
    for model_type in model_results:
        model_results[model_type]['mae'] = pd.concat(model_results[model_type]['mae'], axis=1).mean(axis=1)
        model_results[model_type]['rmse'] = pd.concat(model_results[model_type]['rmse'], axis=1).mean(axis=1)
        model_results[model_type]['ic'] = pd.concat(model_results[model_type]['ic'], axis=1).mean(axis=1)
    
    # Combine predictions across splits
    for pred_type in all_predictions:
        all_predictions[pred_type] = pd.concat(all_predictions[pred_type])
    
    return {
        'model_results': model_results,
        'predictions': all_predictions
    }


if __name__ == "__main__":
    # Example usage
    stock_data, ff_factors, fed_decisions = load_data()
    
    # Calculate returns
    returns = calculate_asset_returns(stock_data)
    
    # Import fed analysis to get regimes
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fed_analysis import identify_fed_regimes
    
    # Get Fed regimes
    fed_with_regimes = identify_fed_regimes(fed_decisions)
    
    # Create a daily regime series
    daily_regime = pd.DataFrame(index=returns.index)
    for date, row in fed_with_regimes.iterrows():
        mask = (daily_regime.index >= date)
        daily_regime.loc[mask, 'regime'] = row['regime']
    
    # Forward fill any missing values
    daily_regime = daily_regime.fillna(method='ffill')
    daily_regime = daily_regime.fillna('Unknown')  # For dates before first Fed decision
    
    # Estimate factor exposures
    factor_exposures = estimate_factor_exposures(returns, ff_factors)
    
    # Calculate factor premiums by regime
    factor_premiums = estimate_factor_premiums_by_regime(ff_factors, daily_regime)
    
    # Plot rolling betas
    fig = plot_rolling_betas(factor_exposures, asset=returns.columns[0])
    plt.savefig(os.path.join(DATA_DIR, 'rolling_betas.png'))
    plt.close(fig)
    
    # Plot factor premiums by regime
    fig = plot_factor_premiums_by_regime(factor_premiums)
    plt.savefig(os.path.join(DATA_DIR, 'factor_premiums_by_regime.png'))
    plt.close(fig)
    
    # Implement time-varying factor model
    model_results = implement_time_varying_factor_model(returns, ff_factors, daily_regime)
    
    print("Factor modeling completed successfully")