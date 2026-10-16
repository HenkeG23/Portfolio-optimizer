import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_efficient_frontier(
    results: pd.DataFrame,
    frontier_volatilities: np.ndarray,
    frontier_returns: np.ndarray,
    opt_sharpe_return: float,
    opt_sharpe_volatility: float,
    opt_min_vol_return: float,
    opt_min_vol_volatility: float
) -> plt.Figure:

    
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_obj = ax.scatter(results['Volatility'], results['Returns'], c=results['Sharpe'],
                              cmap='viridis', marker='o', s=8, alpha=0.25)
    fig.colorbar(scatter_obj, ax=ax, label='Sharpe Ratio')

        # Analytical efficient frontier curve
    ax.plot(frontier_volatilities, frontier_returns, color='black', linewidth=2, label='Efficient Frontier')

    # Only the two optimal points, as discrete markers
    ax.scatter(opt_sharpe_volatility, opt_sharpe_return, color='red', edgecolor='black',
               marker='o', s=150, zorder=5, label='Max Sharpe Portfolio')
    ax.scatter(opt_min_vol_volatility, opt_min_vol_return, color='blue', edgecolor='black',
               marker='o', s=150, zorder=5, label='Min Volatility Portfolio')

    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
    

if __name__ == "__main__":
    from data_fetcher import get_portfolio_data
    from returns import calculate_daily_returns
    from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix, calculate_portfolio_performance 
    from monte_carlo import monte_carlo_simulation
    from optimizer import optimize_max_sharpe, optimize_min_volatility, calculate_efficient_frontier


    # Example usage
    prices = get_portfolio_data(["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"])
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.02)
    frontier_volatilities, frontier_returns = calculate_efficient_frontier(expected_returns.values, cov_matrix.values)

    opt_sharpe_weights = optimize_max_sharpe(expected_returns.values, cov_matrix.values, risk_free_rate=0.02)
    opt_sharpe_return, opt_sharpe_volatility = calculate_portfolio_performance(opt_sharpe_weights, expected_returns, cov_matrix)

    opt_min_vol_weights = optimize_min_volatility(expected_returns.values, cov_matrix.values)
    opt_min_vol_return, opt_min_vol_volatility = calculate_portfolio_performance(opt_min_vol_weights, expected_returns, cov_matrix)

    fig = plot_efficient_frontier(
        results, frontier_volatilities, frontier_returns,
        opt_sharpe_return, opt_sharpe_volatility,
        opt_min_vol_return, opt_min_vol_volatility,
    )

    plt.show()