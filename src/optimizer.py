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

def calculate_efficient_frontier(expected_returns: np.ndarray, cov_matrix: np.ndarray, num_points: int = 50) -> tuple[np.ndarray, np.ndarray]:

    """
    Calculate the efficient frontier by optimizing portfolios for a range of target returns.

    Parameters:
        expected_returns: Array of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
        num_points: Number of points on the efficient frontier to calculate.
    Returns:
        Tuple containing arrays of portfolio volatilities and returns for the efficient frontier.
    """
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    target_returns = np.linspace(min_return, max_return, num_points)

    frontier_volatilities = []
    frontier_returns = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: calculate_portfolio_performance(weights, expected_returns, cov_matrix)[0] - target_return}

        )

        result = minimize(
            portfolio_volatility,
            x0=np.array([1/len(expected_returns)] * len(expected_returns)),
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(len(expected_returns))),
            constraints=constraints
        )

        optimal_weights = result.x
        optimal_return, optimal_volatility = calculate_portfolio_performance(optimal_weights, expected_returns, cov_matrix)
        frontier_volatilities.append(optimal_volatility)
        frontier_returns.append(target_return)

    return np.array(frontier_volatilities), np.array(frontier_returns)

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

    frontier_volatilities, frontier_returns = calculate_efficient_frontier(expected_returns.values, cov_matrix.values)
    print("Efficient Frontier Points:")
    for vol, ret in zip(frontier_volatilities, frontier_returns):
        print(f"Volatility: {vol:.4f}, Return: {ret:.4f}")
        