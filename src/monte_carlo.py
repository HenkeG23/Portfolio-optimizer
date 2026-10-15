import numpy as np
import pandas as pd
from portfolio_metrics import calculate_portfolio_performance

def monte_carlo_simulation(expected_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 10000, risk_free_rate: float = 0.02, max_weight: float = 1.0) -> pd.DataFrame:
    """
    Perform a Monte Carlo simulation to generate random portfolios and calculate their performance.

    Parameters:
        expected_returns: Series of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
        num_portfolios: Number of random portfolios to simulate.
        risk_free_rate: Risk-free rate for Sharpe ratio calculation (default is 0.02).
        max_weight: Maximum weight allowed for any single asset in the portfolio (default is 1.0).
    Returns:
        DataFrame containing portfolio returns, volatilities, and weights for each simulated portfolio.
    """
    num_assets = len(expected_returns)
    

    # Create empty lists to store simulation results
    results = {
        'Returns': [],
        'Volatility': [],
        'Sharpe': [],
        'Weights': []
    }

    max_attempts = num_portfolios * 10
    attempts = 0
    while len(results['Weights']) < num_portfolios and attempts < max_attempts:
        attempts += 1

        # Generate random weights sum up to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Check
        if weights.max() > max_weight:
            continue  # Skip this portfolio 

        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Store results
        results['Returns'].append(portfolio_return)
        results['Volatility'].append(portfolio_volatility)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)
        

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results

def find_optimal_portfolios(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Identify the portfolios with the maximum Sharpe ratio and minimum volatility.

    Parameters:
        results: DataFrame containing portfolio returns, volatilities, and weights for each simulated portfolio.

    Returns:
        Tuple containing the Series of weights for the maximum Sharpe ratio portfolio and the minimum volatility portfolio.
    """

    max_sharpe_idx = results['Sharpe'].idxmax()
    min_volatility_idx = results['Volatility'].idxmin()

    max_sharpe_portfolio = results.loc[max_sharpe_idx]
    min_volatility_portfolio = results.loc[min_volatility_idx]

    return max_sharpe_portfolio, min_volatility_portfolio
    


if __name__ == "__main__":
    from data_fetcher import get_portfolio_data
    from returns import calculate_daily_returns
    from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix

    # Example usage
    prices = get_portfolio_data(["AAPL", "MSFT"])
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.02)

    max_sharpe_portfolio, min_volatility_portfolio = find_optimal_portfolios(results)

    print(results.head())
    print(results.shape)

    print("Maximum Sharpe Ratio Portfolio:")
    print(max_sharpe_portfolio)
    print("Minimum Volatility Portfolio:")
    print(min_volatility_portfolio)


   



