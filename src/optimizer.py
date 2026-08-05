import numpy as np
from scipy.optimize import minimize
from portfolio_metrics import calculate_portfolio_performance


def negative_sharpe(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculate the negative Sharpe ratio for a given set of portfolio weights.

    Parameters:
        weights: Array of portfolio weights (should sum to 1).
        expected_returns: Array of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
        risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    Returns:
        Negative Sharpe ratio (to be minimized).
    """
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # We minimize the negative Sharpe ratio

def optimize_max_sharpe(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> np.ndarray:
    """
    Optimize portfolio weights to maximize the Sharpe ratio.

    Parameters:
        expected_returns: Array of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
        risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    Returns:
        Array of optimized portfolio weights that maximize the Sharpe ratio.
    """
    num_assets = len(expected_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    initial_guess = np.array([1/num_assets] * num_assets)  # Equal weights as initial guess
    results = minimize (
        negative_sharpe,
        x0= initial_guess,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights = results.x

    return optimal_weights

def portfolio_volatility(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate the portfolio volatility for a given set of weights.

    Parameters:
        weights: Array of portfolio weights (should sum to 1).
        expected_returns: Array of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
    Returns:
        Portfolio volatility (standard deviation).
    """
    _, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, cov_matrix)
    return portfolio_volatility

def optimize_min_volatility(expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Optimize portfolio weights to minimize volatility.

    Parameters:
        expected_returns: Array of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
    Returns:
        Array of optimized portfolio weights that minimize volatility.
    """
    num_assets = len(expected_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    initial_guess = np.array([1/num_assets] * num_assets)  # Equal weights as initial guess
    results = minimize(
        portfolio_volatility,
        x0=initial_guess,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights = results.x

    return optimal_weights
    
if __name__ == "__main__":
    from data_fetcher import get_portfolio_data
    from returns import calculate_daily_returns
    from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix

    prices = get_portfolio_data(["AAPL", "MSFT"])
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    optimal_weights = optimize_max_sharpe(expected_returns.values, cov_matrix.values, risk_free_rate=0.02)
    print("Optimal Weights for Maximum Sharpe Ratio Portfolio:", optimal_weights)

    optimal_weights = optimize_min_volatility(expected_returns.values, cov_matrix.values)
    print("Optimal Weights for Minimum Volatility Portfolio:", optimal_weights)
    