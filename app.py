import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

from data_fetcher import get_portfolio_data
from returns import calculate_daily_returns
from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix, calculate_portfolio_performance
from optimizer import optimize_max_sharpe, optimize_min_volatility
from monte_carlo import monte_carlo_simulation, find_optimal_portfolios
from visualization import plot_efficient_frontier
from optimizer import optimize_max_sharpe, optimize_min_volatility, calculate_efficient_frontier

st.title("Portfolio Optimizer")

tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, GOOGL, AMZN, JPM")
risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 5.0, 2.0) / 100  # Convert percentage to decimal

run_button = st.button("Run Optimization")

if run_button:
    st.session_state.analysis_done = True
    st.session_state.tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

if st.session_state.get("analysis_done", False):
    tickers = st.session_state.tickers

    try:
        prices = get_portfolio_data(tickers)
        daily_returns = calculate_daily_returns(prices)

        expected_returns = calculate_expected_returns(daily_returns)
        cov_matrix = calculate_covariance_matrix(daily_returns)
    except ValueError as e:
        st.error(str(e))
        st.stop()
         
    # Max sharpe
    opt_sharpe_weights = optimize_max_sharpe(expected_returns.values, cov_matrix.values, risk_free_rate)
    opt_sharpe_return, opt_sharpe_vol = calculate_portfolio_performance(opt_sharpe_weights, expected_returns, cov_matrix)

    # Min volatility
    opt_minvol_weights = optimize_min_volatility(expected_returns.values, cov_matrix.values)
    opt_minvol_return, opt_minvol_vol = calculate_portfolio_performance(opt_minvol_weights, expected_returns, cov_matrix)

    st.subheader("Maximum Sharpe Ratio Portfolio")
    st.write(f"Expected Return: {opt_sharpe_return:.2%}")
    st.write(f"Volatility: {opt_sharpe_vol:.2%}")
    st.write(f"Sharpe Ratio: {(opt_sharpe_return - risk_free_rate) / opt_sharpe_vol:.3f}")

    weights_df_sharpe = pd.DataFrame({"Ticker": tickers, "Weight": opt_sharpe_weights.round(4)}) 
    st.dataframe(weights_df_sharpe)

    st.subheader("Minimum Volatility Portfolio")
    st.write(f"Expected Return: {opt_minvol_return:.2%}")
    st.write(f"Volatility: {opt_minvol_vol:.2%}")

    weights_df_minvol = pd.DataFrame({"Ticker": tickers, "Weight": opt_minvol_weights.round(4)})
    st.dataframe(weights_df_minvol)

    st.subheader("Efficient Frontier")

    mc_results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=2000, risk_free_rate=risk_free_rate)
    frontier_volatilities, frontier_returns = calculate_efficient_frontier(expected_returns.values, cov_matrix.values, num_points=50)

    fig = plot_efficient_frontier(
        mc_results,
        frontier_volatilities,
        frontier_returns,
        opt_sharpe_return,
        opt_sharpe_vol,
        opt_minvol_return,
        opt_minvol_vol
    )
    st.pyplot(fig)

    st.subheader("Manual Portfolio Weights")
    st.write("Set relative weights - they will be normalized to sum to 1.")

    manual_weights_raw = {}
    for ticker in tickers:
        manual_weights_raw[ticker] = st.slider(f"{ticker}", 0, 100, int(100 / len(tickers)), key=f"weight_{ticker}")

    raw_weights = np.array([manual_weights_raw[t] for t in tickers], dtype=float)

    if np.sum(raw_weights) == 0:
        st.error("At least one weight must be greather than zero")
    else:
        normalized_weights = raw_weights / np.sum(raw_weights)

        manual_return, manual_vol = calculate_portfolio_performance(normalized_weights, expected_returns, cov_matrix)
        manual_sharpe = (manual_return - risk_free_rate) / manual_vol

        st.write(f"Expected Return: {manual_return:.2%}")
        st.write(f"Volatility: {manual_vol:.2%}")
        st.write(f"Sharpe Ratio: {manual_sharpe:.3f}")

        weights_df_manual = pd.DataFrame({
            "Ticker": tickers,
            "Raw Input": raw_weights,
            "Normalized Weight": normalized_weights.round(4)
        })
        st.dataframe(weights_df_manual)

        st.dataframe(weights_df_manual)

        st.subheader("Portfolio Comparison")

        comparison_data = {
            "Portfolio": ["Max Sharpe", "Min Volatility", "Manual"],
            "Expected Return": [opt_sharpe_return * 100, opt_minvol_return * 100, manual_return * 100],
            "Volatility": [opt_sharpe_vol * 100, opt_minvol_vol * 100, manual_vol * 100],
            "Sharpe Ratio": [
                (opt_sharpe_return - risk_free_rate) / opt_sharpe_vol,
                (opt_minvol_return - risk_free_rate) / opt_minvol_vol,
                manual_sharpe
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(
            comparison_df,
            column_config={
                "Expected Return": st.column_config.NumberColumn(format="%.2f%%"),
                "Volatility": st.column_config.NumberColumn(format="%.2f%%"),
                "Sharpe Ratio": st.column_config.NumberColumn(format="%.3f")
            },
            hide_index=True
        )




        

            


