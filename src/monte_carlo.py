import numpy as np
import pandas as pd
from portfolio_metrics import calculate_portfolio_performance

def monte_carlo_simulation(expected_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 10000) -> pd.DataFrame:
    """
    Perform a Monte Carlo simulation to generate random portfolios and calculate their performance.

    Parameters:
        expected_returns: Series of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
        num_portfolios: Number of random portfolios to simulate.
    Returns:
        DataFrame containing portfolio returns, volatilities, and weights for each simulated portfolio.
    """
    num_assets = len(expected_returns)

    # Create empyty lists to store simulation results
    results = {
        'Returns': [],
        'Volatility': [],
        'Weights': []
    }

    i = 0
    for i in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, cov_matrix)

        # Store results
        results['Returns'].append(portfolio_return)
        results['Volatility'].append(portfolio_volatility)
        results['Weights'].append(weights)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    from data_fetcher import get_portfolio_data
    from returns import calculate_daily_returns
    from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix

    # Example usage
    prices = get_portfolio_data(["AAPL", "MSFT"])
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=10000)
    print(results.head())
    print(results.shape)


