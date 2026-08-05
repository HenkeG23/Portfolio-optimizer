import matplotlib.pyplot as plt
import pandas as pd

def plot_efficient_frontier(
        results: pd.DataFrame,
        max_sharpe_portfolio: pd.Series,
        min_volatility_portfolio: pd.Series,
        opt_sharpe_return: float,
        opt_sharpe_volatility: float,
        opt_min_vol_return: float,
        opt_min_vol_volatility: float
        ) -> None:
    """
    Plot the efficient frontier based on Monte Carlo simulation results.

    Parameters:
        results: DataFrame containing portfolio returns, volatilities, and weights for each simulated portfolio.
        max_sharpe_portfolio: Series containing the weights of the portfolio with the maximum Sharpe ratio.
        min_volatility_portfolio: Series containing the weights of the portfolio with the minimum volatility.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results['Volatility'], results['Returns'], c=results['Sharpe'], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Returns'], color='r', marker='*', s=500, label='Max Sharpe Ratio')
    plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Returns'], color='b', marker='*', s=500, label='Min Volatility')

    plt.scatter(opt_sharpe_volatility, opt_sharpe_return, color='orange', marker='X', s=500, label='Optimal Sharpe Portfolio')
    plt.scatter(opt_min_vol_volatility, opt_min_vol_return, color='green', marker='X', s=500, label='Optimal Min Volatility Portfolio')

    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    from data_fetcher import get_portfolio_data
    from returns import calculate_daily_returns
    from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix, calculate_portfolio_performance 
    from monte_carlo import monte_carlo_simulation, find_optimal_portfolios
    from optimizer import optimize_max_sharpe, optimize_min_volatility


    # Example usage
    # prices = get_portfolio_data(["AAPL", "MSFT"])
    prices = get_portfolio_data(["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"])
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.02)

    max_sharpe_portfolio, min_volatility_portfolio = find_optimal_portfolios(results)

    opt_sharpe_weights = optimize_max_sharpe(expected_returns.values, cov_matrix.values, risk_free_rate=0.02)
    opt_sharpe_return, opt_sharpe_volatility = calculate_portfolio_performance(opt_sharpe_weights, expected_returns, cov_matrix)

    opt_min_vol_weights = optimize_min_volatility(expected_returns.values, cov_matrix.values)
    opt_min_vol_return, opt_min_vol_volatility = calculate_portfolio_performance(opt_min_vol_weights, expected_returns, cov_matrix)

    plot_efficient_frontier(
        results,
        max_sharpe_portfolio,
        min_volatility_portfolio,
        opt_sharpe_return,
        opt_sharpe_volatility,
        opt_min_vol_return,
        opt_min_vol_volatility
    )
