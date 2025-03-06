"""
Portfolio Optimization Module

This module implements portfolio optimization techniques that incorporate:
1. Mean-variance optimization with factor-based expected returns
2. Fed regime-conditional optimization
3. Backtesting framework to evaluate optimization strategies
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import cvxpy as cp
from sklearn.covariance import LedoitWolf, OAS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


def load_data():
    """
    Load processed data for portfolio optimization
    
    Returns:
    --------
    tuple
        Tuple containing (stock_data, ff_factors, fed_decisions, fed_regimes)
    """
    logger.info("Loading processed data for portfolio optimization")
    
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
        
        # Import fed analysis functions
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.fed_analysis import identify_fed_regimes
        
        # Get Fed regimes
        fed_regimes = identify_fed_regimes(fed_decisions)
        
        logger.info("Data loaded successfully")
        return stock_data, ff_factors, fed_decisions, fed_regimes
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_daily_regime_data(asset_returns, fed_regimes):
    """
    Create a daily series of Fed regimes aligned with asset returns
    
    Parameters:
    -----------
    asset_returns : pandas.DataFrame
        DataFrame containing asset returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily returns and regime data
    """
    logger.info("Creating daily regime data")
    
    # Create a daily regime series
    daily_regime = pd.DataFrame(index=asset_returns.index)
    
    # Assign regimes to each day based on the most recent Fed decision
    for date, row in fed_regimes.iterrows():
        mask = (daily_regime.index >= date)
        daily_regime.loc[mask, 'regime'] = row['regime']
    
    # Forward fill any missing values
    daily_regime = daily_regime.fillna(method='ffill')
    daily_regime = daily_regime.fillna('Unknown')  # For dates before first Fed decision
    
    # Combine with returns
    combined_data = pd.concat([asset_returns, daily_regime], axis=1)
    
    return combined_data


def estimate_covariance(returns, method='ledoit-wolf', window=None):
    """
    Estimate covariance matrix of returns using various methods
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    method : str, optional
        Covariance estimation method, default is 'ledoit-wolf'
    window : int, optional
        Window size for rolling estimation, if None uses all data
        
    Returns:
    --------
    pandas.DataFrame
        Covariance matrix as DataFrame
    """
    logger.info(f"Estimating covariance using {method} method")
    
    if window is not None:
        # Implement rolling window covariance
        rolling_cov = returns.rolling(window=window).cov().dropna()
        # This returns a MultiIndex DataFrame, we need to take the last window
        last_window_idx = returns.index[-window:]
        cov_matrix = rolling_cov.loc[last_window_idx[-1]]
        
        # Convert to DataFrame with proper index/columns
        assets = returns.columns
        cov_df = pd.DataFrame(
            [[cov_matrix.loc[(last_window_idx[-1], a1), a2] for a2 in assets] for a1 in assets],
            index=assets, columns=assets
        )
        
        return cov_df
    
    if method == 'sample':
        # Sample covariance
        cov_matrix = returns.cov()
    
    elif method == 'ledoit-wolf':
        # Ledoit-Wolf shrinkage
        lw = LedoitWolf().fit(returns)
        cov_matrix = pd.DataFrame(
            lw.covariance_, 
            index=returns.columns, 
            columns=returns.columns
        )
    
    elif method == 'oas':
        # Oracle Approximating Shrinkage
        oas = OAS().fit(returns)
        cov_matrix = pd.DataFrame(
            oas.covariance_, 
            index=returns.columns, 
            columns=returns.columns
        )
    
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    
    return cov_matrix


def mean_variance_optimization(expected_returns, cov_matrix, target_return=None, target_risk=None, 
                              risk_aversion=1, constraints=None):
    """
    Perform mean-variance portfolio optimization
    
    Parameters:
    -----------
    expected_returns : pandas.Series
        Series containing expected returns for each asset
    cov_matrix : pandas.DataFrame
        Covariance matrix of asset returns
    target_return : float, optional
        Target portfolio return, if None maximize Sharpe ratio
    target_risk : float, optional
        Target portfolio risk, if None maximize Sharpe ratio
    risk_aversion : float, optional
        Risk aversion parameter for utility maximization
    constraints : dict, optional
        Additional constraints for the optimization
        
    Returns:
    --------
    pandas.Series
        Series containing optimal weights for each asset
    """
    logger.info("Performing mean-variance optimization")
    
    n_assets = len(expected_returns)
    assets = expected_returns.index
    
    # Define optimization variables
    w = cp.Variable(n_assets)
    
    # Objective function depends on the problem formulation
    if target_return is not None:
        # Minimize risk subject to target return
        objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))
        constraints_list = [
            cp.sum(w) == 1,                                   # Fully invested
            w >= 0,                                           # Long-only
            expected_returns.values @ w >= target_return      # Return target
        ]
    
    elif target_risk is not None:
        # Maximize return subject to target risk
        objective = cp.Maximize(expected_returns.values @ w)
        constraints_list = [
            cp.sum(w) == 1,                                   # Fully invested
            w >= 0,                                           # Long-only
            cp.sqrt(cp.quad_form(w, cov_matrix.values)) <= target_risk  # Risk constraint
        ]
    
    else:
        # Maximize utility (return - risk_aversion * variance)
        objective = cp.Maximize(expected_returns.values @ w - risk_aversion * cp.quad_form(w, cov_matrix.values))
        constraints_list = [
            cp.sum(w) == 1,                                   # Fully invested
            w >= 0                                            # Long-only
        ]
    
    # Add custom constraints if provided
    if constraints:
        if 'factor_exposures' in constraints:
            # Add factor exposure constraints
            factor_exposures = constraints['factor_exposures']
            factor_targets = constraints['factor_targets']
            
            for factor, target in factor_targets.items():
                if factor in factor_exposures:
                    factor_beta = factor_exposures[factor].values
                    constraints_list.append(factor_beta @ w == target)
        
        if 'max_weight' in constraints:
            # Add maximum weight constraint
            max_weight = constraints['max_weight']
            constraints_list.append(w <= max_weight)
        
        if 'min_weight' in constraints:
            # Add minimum weight constraint
            min_weight = constraints['min_weight']
            constraints_list.append(w >= min_weight)
    
    # Solve the optimization problem
    prob = cp.Problem(objective, constraints_list)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        try:
            prob.solve(solver=cp.SCS)
        except:
            logger.error("Optimization failed")
            return None
    
    if prob.status != 'optimal':
        logger.warning(f"Optimization status: {prob.status}")
    
    # Extract optimal weights
    weights = pd.Series(w.value, index=assets)
    
    # Calculate portfolio statistics
    portfolio_return = (weights * expected_returns).sum()
    portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    logger.info(f"Optimization results: Return={portfolio_return:.4f}, Risk={portfolio_risk:.4f}, Sharpe={sharpe_ratio:.4f}")
    
    # Check weights sum to 1 (within tolerance)
    if not np.isclose(weights.sum(), 1.0, atol=1e-4):
        logger.warning(f"Weights sum to {weights.sum():.6f}, not 1.0")
        # Normalize weights
        weights = weights / weights.sum()
    
    return weights


def optimize_by_regime(returns_with_regime, expected_returns_by_regime, 
                       window=252, rebalance_freq=21, method='risk_parity'):
    """
    Optimize portfolios conditional on Fed regime
    
    Parameters:
    -----------
    returns_with_regime : pandas.DataFrame
        DataFrame containing asset returns and regime data
    expected_returns_by_regime : dict
        Dictionary mapping regimes to expected returns Series
    window : int, optional
        Window size for covariance estimation
    rebalance_freq : int, optional
        Frequency of portfolio rebalancing in days
    method : str, optional
        Optimization method, options are 'mean_variance', 'min_variance', 'risk_parity'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing optimal weights by date and regime
    """
    logger.info(f"Optimizing portfolios by regime using {method} method")
    
    returns = returns_with_regime.drop('regime', axis=1)
    regimes = returns_with_regime['regime']
    
    # Get unique regimes
    unique_regimes = regimes.unique()
    
    # Initialize results
    weights_by_date = {}
    
    # Define rebalancing dates
    all_dates = returns.index
    rebalance_dates = []
    
    for i in range(window, len(all_dates), rebalance_freq):
        rebalance_dates.append(all_dates[i])
    
    # For each rebalancing date
    for date in rebalance_dates:
        # Get historical data for the estimation window
        window_data = returns.loc[:date].iloc[-window:]
        current_regime = regimes.loc[date]
        
        # Estimate covariance matrix
        cov_matrix = estimate_covariance(window_data)
        
        # Get expected returns for current regime
        if current_regime in expected_returns_by_regime:
            expected_returns = expected_returns_by_regime[current_regime]
        else:
            # Use average of all regimes if current regime is not available
            all_expected_returns = pd.concat(expected_returns_by_regime.values(), axis=1)
            expected_returns = all_expected_returns.mean(axis=1)
        
        # Perform optimization based on selected method
        if method == 'mean_variance':
            # Traditional mean-variance optimization
            weights = mean_variance_optimization(
                expected_returns, 
                cov_matrix, 
                risk_aversion=1
            )
        
        elif method == 'min_variance':
            # Minimum variance portfolio
            ones = pd.Series(1, index=cov_matrix.index)
            weights = mean_variance_optimization(
                ones,  # Dummy expected returns (all 1s)
                cov_matrix,
                constraints={'max_weight': 0.2}  # Limit concentration
            )
        
        elif method == 'risk_parity':
            # Risk parity implementation
            
            def risk_contribution(weights, cov_matrix):
                # Calculate portfolio risk
                portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
                
                # Calculate marginal risk contribution of each asset
                marginal_contrib = cov_matrix @ weights / portfolio_risk
                
                # Calculate risk contribution
                risk_contrib = weights * marginal_contrib
                
                return risk_contrib
            
            def risk_parity_objective(weights, cov_matrix):
                weights = np.array(weights).reshape(-1)
                
                # Calculate risk contribution
                risk_contrib = risk_contribution(weights, cov_matrix)
                
                # Target equal risk contribution
                target_risk_contrib = np.ones_like(weights) * np.sum(risk_contrib) / len(weights)
                
                # Minimize sum of squared differences
                return np.sum((risk_contrib - target_risk_contrib)**2)
            
            n_assets = len(cov_matrix)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints: weights sum to 1 and all weights are positive
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                          {'type': 'ineq', 'fun': lambda w: w})
            
            # Minimize risk parity objective
            opt_result = minimize(
                risk_parity_objective,
                initial_weights,
                args=(cov_matrix.values,),
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 0.3) for _ in range(n_assets)]  # Limit concentration
            )
            
            if not opt_result.success:
                logger.warning(f"Risk parity optimization failed for date {date}")
                weights = pd.Series(initial_weights, index=cov_matrix.index)
            else:
                weights = pd.Series(opt_result.x, index=cov_matrix.index)
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store weights
        weights_by_date[date] = weights
    
    # Combine weights into DataFrame
    all_weights = pd.DataFrame(weights_by_date).T
    
    # Add regime information
    all_weights['regime'] = regimes.loc[all_weights.index]
    
    return all_weights


def calculate_portfolio_returns(weights, returns, rebalance_freq=21):
    """
    Calculate portfolio returns based on weights and asset returns
    
    Parameters:
    -----------
    weights : pandas.DataFrame
        DataFrame containing portfolio weights by date
    returns : pandas.DataFrame
        DataFrame containing asset returns
    rebalance_freq : int, optional
        Frequency of portfolio rebalancing in days
        
    Returns:
    --------
    pandas.Series
        Series containing portfolio returns
    """
    logger.info("Calculating portfolio returns")
    
    # Extract asset columns (excluding 'regime' if present)
    asset_columns = weights.columns.drop('regime') if 'regime' in weights.columns else weights.columns
    
    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    
    # Get rebalancing dates
    rebalance_dates = weights.index
    
    # For each period between rebalancing dates
    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]
        
        # Get weights for this period
        period_weights = weights.loc[start_date, asset_columns]
        
        # Get returns for this period
        period_returns = returns.loc[start_date:end_date, asset_columns]
        
        # Calculate portfolio returns
        period_portfolio_returns = (period_returns * period_weights).sum(axis=1)
        
        # Store in the result series
        portfolio_returns.loc[start_date:end_date] = period_portfolio_returns
    
    # Handle the last period
    if len(rebalance_dates) > 0:
        last_date = rebalance_dates[-1]
        last_weights = weights.loc[last_date, asset_columns]
        last_returns = returns.loc[last_date:, asset_columns]
        last_portfolio_returns = (last_returns * last_weights).sum(axis=1)
        portfolio_returns.loc[last_date:] = last_portfolio_returns
    
    return portfolio_returns


def calculate_performance_metrics(returns, risk_free_rate=0):
    """
    Calculate performance metrics for a returns series
    
    Parameters:
    -----------
    returns : pandas.Series
        Series containing portfolio returns
    risk_free_rate : float, optional
        Risk-free rate (daily)
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    logger.info("Calculating performance metrics")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        logger.warning("No valid returns data to calculate metrics")
        return {}
    
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Calculate volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cum_returns = (1 + returns).cumprod()
    drawdowns = cum_returns / cum_returns.cummax() - 1
    max_drawdown = drawdowns.min()
    
    # Calculate Sortino ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (returns.mean() - risk_free_rate) * 252 / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Calmar ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    # Calculate average win/loss
    avg_win = returns[returns > 0].mean() if any(returns > 0) else 0
    avg_loss = returns[returns < 0].mean() if any(returns < 0) else 0
    
    # Return results
    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }
    
    return metrics


def plot_performance_comparison(strategies_returns, benchmark_returns=None):
    """
    Plot performance comparison of different strategies
    
    Parameters:
    -----------
    strategies_returns : dict
        Dictionary mapping strategy names to returns Series
    benchmark_returns : pandas.Series, optional
        Series containing benchmark returns
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    logger.info("Plotting performance comparison")
    
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Add benchmark to strategies if provided
    all_returns = strategies_returns.copy()
    if benchmark_returns is not None:
        all_returns['Benchmark'] = benchmark_returns
    
    # Calculate cumulative returns
    cumulative_returns = {}
    drawdowns = {}
    rolling_volatility = {}
    
    for name, returns in all_returns.items():
        # Calculate cumulative returns
        cumulative_returns[name] = (1 + returns).cumprod()
        
        # Calculate drawdowns
        cum_ret = cumulative_returns[name]
        drawdowns[name] = cum_ret / cum_ret.cummax() - 1
        
        # Calculate rolling volatility (annualized, 63-day window)
        rolling_volatility[name] = returns.rolling(63).std() * np.sqrt(252)
    
    # Plot cumulative returns
    for name, cum_ret in cumulative_returns.items():
        axes[0].plot(cum_ret.index, cum_ret, label=name)
    
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('Strategy Performance Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot drawdowns
    for name, dd in drawdowns.items():
        axes[1].plot(dd.index, dd, label=name)
    
    axes[1].set_ylabel('Drawdown')
    axes[1].fill_between(drawdowns[list(drawdowns.keys())[0]].index, 0, -1, color='red', alpha=0.1)
    axes[1].grid(True)
    
    # Plot rolling volatility
    for name, vol in rolling_volatility.items():
        axes[2].plot(vol.index, vol, label=name)
    
    axes[2].set_ylabel('Rolling Volatility (63d)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    return fig


def backtest_regime_strategies(returns, fed_regimes, factor_data=None, rebalance_freq=21, 
                              window=252, benchmark_ticker='SPY'):
    """
    Backtest investment strategies conditional on Fed regimes
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
    factor_data : pandas.DataFrame, optional
        DataFrame containing factor data
    rebalance_freq : int, optional
        Frequency of portfolio rebalancing in days
    window : int, optional
        Window size for estimation
    benchmark_ticker : str, optional
        Ticker symbol for benchmark
        
    Returns:
    --------
    dict
        Dictionary containing backtest results
    """
    logger.info("Backtesting regime-based strategies")
    
    # Create daily regime data
    returns_with_regime = create_daily_regime_data(returns, fed_regimes)
    
    # Define strategies to test
    strategies = {
        'Equal Weight': 'equal_weight',
        'Minimum Variance': 'min_variance',
        'Risk Parity': 'risk_parity',
        'Regime-Based': 'regime_based'
    }
    
    # Calculate expected returns by regime
    expected_returns_by_regime = {}
    
    for regime, group in returns_with_regime.groupby('regime'):
        # Extract returns for this regime
        regime_returns = group.drop('regime', axis=1)
        
        # Calculate expected returns
        expected_returns_by_regime[regime] = regime_returns.mean() * 252  # Annualized
    
    # Calculate portfolio weights for each strategy
    strategy_weights = {}
    
    # Get asset columns
    assets = returns.columns
    
    # Define rebalancing dates
    all_dates = returns.index
    rebalance_dates = []
    
    for i in range(window, len(all_dates), rebalance_freq):
        rebalance_dates.append(all_dates[i])
    
    # Equal Weight strategy
    equal_weights = pd.DataFrame(
        {date: pd.Series(1/len(assets), index=assets) for date in rebalance_dates}
    ).T
    strategy_weights['equal_weight'] = equal_weights
    
    # Minimum Variance strategy
    min_var_weights = optimize_by_regime(
        returns_with_regime, 
        expected_returns_by_regime, 
        window=window, 
        rebalance_freq=rebalance_freq, 
        method='min_variance'
    )
    strategy_weights['min_variance'] = min_var_weights
    
    # Risk Parity strategy
    rp_weights = optimize_by_regime(
        returns_with_regime, 
        expected_returns_by_regime, 
        window=window, 
        rebalance_freq=rebalance_freq, 
        method='risk_parity'
    )
    strategy_weights['risk_parity'] = rp_weights
    
    # Regime-Based strategy (use mean-variance optimization conditional on regime)
    # We'll implement a simple approach: use different risk aversion by regime
    regime_based_weights = {}
    
    for date in rebalance_dates:
        # Get historical data for the estimation window
        window_data = returns.loc[:date].iloc[-window:]
        current_regime = returns_with_regime.loc[date, 'regime']
        
        # Estimate covariance matrix
        cov_matrix = estimate_covariance(window_data)
        
        # Get expected returns for current regime
        if current_regime in expected_returns_by_regime:
            expected_returns = expected_returns_by_regime[current_regime]
        else:
            # Use average of all regimes if current regime is not available
            all_expected_returns = pd.concat(expected_returns_by_regime.values(), axis=1)
            expected_returns = all_expected_returns.mean(axis=1)
        
        # Adjust risk aversion based on regime
        if current_regime == 'Tightening':
            risk_aversion = 2.0  # More conservative in tightening regime
        elif current_regime == 'Easing':
            risk_aversion = 0.5  # More aggressive in easing regime
        else:
            risk_aversion = 1.0  # Neutral
        
        # Perform optimization
        weights = mean_variance_optimization(
            expected_returns, 
            cov_matrix, 
            risk_aversion=risk_aversion,
            constraints={'max_weight': 0.3}  # Limit concentration
        )
        
        regime_based_weights[date] = weights
    
    # Combine weights into DataFrame
    regime_based_df = pd.DataFrame(regime_based_weights).T
    
    # Add regime information
    regime_based_df['regime'] = returns_with_regime.loc[regime_based_df.index, 'regime']
    
    strategy_weights['regime_based'] = regime_based_df
    
    # Calculate portfolio returns for each strategy
    strategy_returns = {}
    
    for name, strategy in strategies.items():
        weights = strategy_weights[strategy]
        strategy_returns[name] = calculate_portfolio_returns(weights, returns, rebalance_freq)
    
    # Get benchmark returns if available
    benchmark_returns = None
    if benchmark_ticker in returns.columns:
        benchmark_returns = returns[benchmark_ticker]
    
    # Calculate performance metrics
    performance_metrics = {}
    
    for name, returns_series in strategy_returns.items():
        performance_metrics[name] = calculate_performance_metrics(returns_series)
    
    # Add benchmark metrics if available
    if benchmark_returns is not None:
        performance_metrics['Benchmark'] = calculate_performance_metrics(benchmark_returns)
    
    # Plot performance comparison
    fig = plot_performance_comparison(strategy_returns, benchmark_returns)
    
    # Return results
    results = {
        'strategy_weights': strategy_weights,
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'performance_metrics': performance_metrics,
        'expected_returns_by_regime': expected_returns_by_regime,
        'figure': fig
    }
    
    return results


def create_factor_timing_portfolio(returns, factor_data, fed_regimes, factor_timing_signals):
    """
    Create a portfolio that times factor exposures based on Fed regime
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns
    factor_data : pandas.DataFrame
        DataFrame containing factor returns
    fed_regimes : pandas.DataFrame
        DataFrame containing Fed regime classifications
    factor_timing_signals : dict
        Dictionary mapping factors to timing signals
        
    Returns:
    --------
    dict
        Dictionary containing portfolio results
    """
    logger.info("Creating factor timing portfolio")
    
    # Create daily regime data
    returns_with_regime = create_daily_regime_data(returns, fed_regimes)
    
    # Calculate factor exposures for each asset
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.factor_modeling import estimate_factor_exposures
    
    factor_exposures = estimate_factor_exposures(returns, factor_data)
    
    # Get the most recent factor exposures
    recent_exposures = factor_exposures['betas'].iloc[-1].unstack(level=1)
    
    # Create portfolio for each factor based on timing signals
    factor_portfolios = {}
    
    for factor, signal in factor_timing_signals.items():
        # Skip if factor is not in our data
        if factor not in recent_exposures.columns:
            logger.warning(f"Factor {factor} not found in exposures data")
            continue
        
        # Get exposures for this factor
        factor_betas = recent_exposures[factor]
        
        # Sort assets by factor exposure
        sorted_assets = factor_betas.sort_values()
        
        # Determine long/short based on signal
        if signal > 0:  # Positive signal -> long high exposure, short low exposure
            long_assets = sorted_assets.iloc[-5:]  # Top 5 by exposure
            short_assets = sorted_assets.iloc[:5]  # Bottom 5 by exposure
        else:  # Negative signal -> long low exposure, short high exposure
            long_assets = sorted_assets.iloc[:5]  # Bottom 5 by exposure
            short_assets = sorted_assets.iloc[-5:]  # Top 5 by exposure
        
        # Equal weight within long and short
        weights = pd.Series(0, index=returns.columns)
        weights[long_assets.index] = 1 / len(long_assets)
        weights[short_assets.index] = -1 / len(short_assets)
        
        factor_portfolios[factor] = weights
    
    # Combine factor portfolios with equal weight
    combined_weights = pd.DataFrame(factor_portfolios)
    portfolio_weights = combined_weights.mean(axis=1)
    
    # Normalize to long-only if needed
    if portfolio_weights.min() < 0:
        portfolio_weights = portfolio_weights - portfolio_weights.min()
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
    
    # Calculate portfolio returns
    portfolio_returns = (returns * portfolio_weights).sum(axis=1)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns)
    
    # Return results
    results = {
        'weights': portfolio_weights,
        'returns': portfolio_returns,
        'metrics': metrics,
        'factor_portfolios': factor_portfolios
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    stock_data, ff_factors, fed_decisions, fed_regimes = load_data()
    
    # Calculate returns
    returns = stock_data.pct_change().dropna()
    
    # Backtest regime strategies
    backtest_results = backtest_regime_strategies(
        returns, 
        fed_regimes, 
        factor_data=ff_factors, 
        rebalance_freq=21
    )
    
    # Save performance plot
    backtest_results['figure'].savefig(os.path.join(DATA_DIR, 'regime_strategies_performance.png'))
    plt.close(backtest_results['figure'])
    
    # Create sample factor timing signals
    # In a real implementation, these would come from the fed_analysis module
    factor_timing_signals = {
        'mkt_excess': 1,  # Positive on market
        'smb': -1,        # Negative on size (prefer large caps)
        'hml': 1          # Positive on value
    }
    
    # Create factor timing portfolio
    timing_results = create_factor_timing_portfolio(
        returns, 
        ff_factors, 
        fed_regimes, 
        factor_timing_signals
    )
    
    print("Portfolio optimization completed successfully")
    
    # Print summary of results
    print("\nPerformance Metrics:")
    metrics_df = pd.DataFrame(backtest_results['performance_metrics']).T
    print(metrics_df[['total_return', 'cagr', 'volatility', 'sharpe_ratio', 'max_drawdown']].round(4))
    
    print("\nFactor Timing Portfolio Performance:")
    print(pd.Series(timing_results['metrics']).round(4))