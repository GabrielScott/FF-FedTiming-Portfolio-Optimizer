# FF-FedTiming-Portfolio-Optimizer

A Python-based portfolio optimization tool that integrates Fama-French factor analysis with Federal Reserve policy timing to generate enhanced investment strategies.

## Overview

This project implements a comprehensive portfolio optimization framework that combines the Fama-French three-factor model with Federal Reserve monetary policy analysis. The goal is to develop strategies that can capture additional alpha by timing factor exposures based on Fed policy regimes.

Key capabilities:
- Fetch and analyze stock price data alongside Fama-French factors
- Identify Federal Reserve policy regimes (Tightening, Easing, Neutral)
- Calculate time-varying factor exposures and factor premiums by regime
- Optimize portfolios using various approaches (mean-variance, risk parity)
- Implement regime-conditional investment strategies
- Visualize and analyze portfolio performance with interactive dashboards

## Project Structure

```
FF-FedTiming-Portfolio-Optimizer/
│
├── data/                          # Data storage directory
│   ├── raw/                       # Raw downloaded data
│   ├── processed/                 # Processed datasets
│   └── visualizations/            # Saved figures and dashboards
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_collection.py         # Fetches stock data and FF factors
│   ├── fed_analysis.py            # Analyzes Fed rate decisions
│   ├── factor_modeling.py         # Calculates factor exposures
│   ├── optimization.py            # Implements portfolio optimization
│   └── visualization.py           # Creates visualization dashboards
│
├── notebooks/                     # Jupyter notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_factor_analysis.ipynb
│   ├── 3_fed_impact_study.ipynb
│   └── 4_portfolio_optimization.ipynb
│
├── tests/                         # Unit tests
│
├── requirements.txt               # Project dependencies
├── setup.py                       # Installation script
└── README.md                      # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FF-FedTiming-Portfolio-Optimizer.git
cd FF-FedTiming-Portfolio-Optimizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Workflow

1. **Collect Data**:
```python
from src.data_collection import fetch_stock_data, fetch_fama_french_factors, fetch_fed_rate_decisions

# Fetch historical stock data
tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
start_date = '2010-01-01'
end_date = '2023-12-31'
stock_data = fetch_stock_data(tickers, start_date, end_date)

# Fetch Fama-French factors
ff_factors = fetch_fama_french_factors(frequency='daily')

# Fetch Fed rate decisions
fed_decisions = fetch_fed_rate_decisions()
```

2. **Analyze Fed Regimes**:
```python
from src.fed_analysis import identify_fed_regimes, analyze_returns_by_regime

# Identify Fed policy regimes
fed_regimes = identify_fed_regimes(fed_decisions)

# Analyze asset returns by regime
regime_stats = analyze_returns_by_regime(stock_data, fed_regimes)
```

3. **Calculate Factor Exposures**:
```python
from src.factor_modeling import calculate_asset_returns, estimate_factor_exposures

# Calculate asset returns
returns = calculate_asset_returns(stock_data)

# Estimate factor exposures
factor_exposures = estimate_factor_exposures(returns, ff_factors)
```

4. **Optimize Portfolio**:
```python
from src.optimization import create_daily_regime_data, backtest_regime_strategies

# Create regime-based data
returns_with_regime = create_daily_regime_data(returns, fed_regimes)

# Backtest regime-conditional strategies
backtest_results = backtest_regime_strategies(
    returns, 
    fed_regimes, 
    factor_data=ff_factors,
    rebalance_freq=21  # Monthly rebalancing
)
```

5. **Visualize Results**:
```python
from src.visualization import create_interactive_regime_dashboard, create_portfolio_performance_dashboard

# Create interactive dashboards
regime_dashboard = create_interactive_regime_dashboard(returns, fed_regimes, factor_exposures)
performance_dashboard = create_portfolio_performance_dashboard(
    backtest_results['strategy_returns'],
    backtest_results['strategy_weights'],
    backtest_results['benchmark_returns']
)
```

### Running from Jupyter Notebooks

For a more interactive experience, explore the Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter notebook notebooks/
```

## Key Features

### Fama-French Factor Analysis

- Uses the classic three-factor model (Market, Size, Value)
- Calculates time-varying factor exposures using rolling regressions
- Estimates factor premiums conditional on Federal Reserve policy regimes

### Fed Policy Regime Classification

- Identifies monetary policy regimes (Tightening, Easing, Neutral)
- Analyzes market behavior around Fed decision dates
- Creates trading signals based on regime changes and Fed surprises

### Portfolio Optimization Methods

- Mean-variance optimization with regime-based expected returns
- Minimum variance portfolios with concentration limits
- Risk parity optimization for balanced risk contribution
- Regime-conditional strategy with dynamic risk aversion

### Interactive Visualizations

- Regime dashboards showing asset performance across policy environments
- Factor exposure heatmaps and time series
- Portfolio performance metrics and comparison tools
- Regime transition probability analysis

## Example Results

Here are some key insights this analysis framework can generate:

1. **Factor Performance by Fed Regime**:
   - Value factor (HML) tends to outperform during tightening cycles
   - Small-cap stocks (SMB) typically excel during easing regimes
   - Market risk premium varies significantly based on Fed policy direction

2. **Regime-Based Strategy Performance**:
   - Strategies that adjust risk exposure based on regime tend to have better risk-adjusted returns
   - Proper factor timing can add significant alpha over static allocations
   - Combining regime signals with factor exposures improves Sharpe ratio

3. **Portfolio Optimization Benefits**:
   - Regime-conditional risk management reduces maximum drawdowns
   - Factor-based optimization increases diversification benefits
   - Dynamic weighting schemes outperform static allocations

## Future Enhancements

Potential areas for further development:

1. Incorporate additional factors (Momentum, Quality, Low Volatility)
2. Add machine learning models for regime prediction
3. Implement alternative risk measures (CVaR, Expected Shortfall)
4. Develop options-based hedging strategies for regime transitions
5. Integrate natural language processing of Fed minutes and statements

## Dependencies

- [pandas](https://pandas.pydata.org/): Data manipulation and analysis
- [numpy](https://numpy.org/): Numerical computing
- [matplotlib](https://matplotlib.org/): Static visualizations
- [seaborn](https://seaborn.pydata.org/): Statistical visualizations
- [plotly](https://plotly.com/python/): Interactive dashboards
- [yfinance](https://pypi.org/project/yfinance/): Yahoo Finance data access
- [scikit-learn](https://scikit-learn.org/): Machine learning utilities
- [statsmodels](https://www.statsmodels.org/): Statistical models and tests
- [cvxpy](https://www.cvxpy.org/): Convex optimization framework
- [beautifulsoup4](https://pypi.org/project/beautifulsoup4/): Web scraping

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kenneth French for providing the factor data
- Federal Reserve Economic Data (FRED) for economic indicators
- Open-source community for the excellent libraries used in this project

## Contact

For questions or feedback, please reach out to [your email or contact information].
