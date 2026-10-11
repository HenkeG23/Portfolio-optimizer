import streamlit as st
import sys
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

from data_fetcher import get_portfolio_data, search_tickers, years_ago
from returns import calculate_daily_returns
from portfolio_metrics import calculate_expected_returns, calculate_covariance_matrix, calculate_portfolio_performance, calculate_buy_and_hold_value
from monte_carlo import monte_carlo_simulation
from visualization import plot_efficient_frontier
from optimizer import optimize_max_sharpe, optimize_min_volatility, calculate_efficient_frontier

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
PERIODS = {"1 year": 1, "3 years": 3, "5 years": 5, "10 years": 10, "Custom": None}
CURRENCIES = ["SEK", "USD", "EUR"]
BENCHMARKS = {"OMXS30": "^OMX", "S&P 500": "^GSPC", "None": None}
INITIAL_VALUE = 10_000


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str) -> list[dict]:
    return search_tickers(query)


@st.cache_data(ttl=3600, show_spinner="Fetching data and optimizing...")
def run_analysis(tickers: tuple, start: str, end: str, base_currency: str, risk_free_rate: float, max_weight: float) -> dict:
    """Fetch prices and run all optimizations. Cached so moving the manual weight sliders stays fast."""
    prices = get_portfolio_data(list(tickers), start, end, base_currency)
    daily_returns = calculate_daily_returns(prices)

    expected_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    sharpe_weights = optimize_max_sharpe(expected_returns.values, cov_matrix.values, risk_free_rate, max_weight)
    minvol_weights = optimize_min_volatility(expected_returns.values, cov_matrix.values, max_weight)

    mc_results = monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=2000, risk_free_rate=risk_free_rate)
    frontier_volatilities, frontier_returns = calculate_efficient_frontier(
        expected_returns.values, cov_matrix.values, num_points=50, max_weight=max_weight
    )

    return {
        "prices": prices,
        "expected_returns": expected_returns,
        "cov_matrix": cov_matrix,
        "sharpe_weights": sharpe_weights,
        "minvol_weights": minvol_weights,
        "mc_results": mc_results,
        "frontier_volatilities": frontier_volatilities,
        "frontier_returns": frontier_returns,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(symbol: str, start: str, end: str, base_currency: str) -> pd.Series:
    return get_portfolio_data([symbol], start, end, base_currency)[symbol]


def add_selected_to_portfolio():
    for symbol in st.session_state.search_pick:
        if symbol not in st.session_state.portfolio:
            st.session_state.portfolio.append(symbol)
    st.session_state.portfolio_select = st.session_state.portfolio.copy()
    st.session_state.search_pick = []


def sync_portfolio():
    st.session_state.portfolio = st.session_state.portfolio_select.copy()


if "portfolio" not in st.session_state:
    st.session_state.portfolio = DEFAULT_TICKERS.copy()
    st.session_state.portfolio_select = DEFAULT_TICKERS.copy()

# Settings
st.sidebar.header("Settings")

period = st.sidebar.selectbox("Period", list(PERIODS), index=2)
end = date.today().isoformat()
if PERIODS[period] is None:
    date_range = st.sidebar.date_input(
        "Date range",
        value=(date.fromisoformat(years_ago(5)), date.today()),
        max_value=date.today(),
    )
    if len(date_range) != 2:
        st.sidebar.info("Pick an end date.")
        st.stop()
    start, end = date_range[0].isoformat(), date_range[1].isoformat()
else:
    start = years_ago(PERIODS[period])

base_currency = st.sidebar.selectbox(
    "Currency", CURRENCIES, help="All prices are converted to this currency before the analysis."
)
risk_free_rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 5.0, 2.0) / 100  # Convert percentage to decimal
max_weight = st.sidebar.slider(
    "Max weight per stock (%)", 5, 100, 100, step=5,
    help="Upper limit for any single stock in the optimized portfolios."
) / 100
benchmark_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS))

st.title("Portfolio Optimizer")

query = st.text_input("Search stocks by name or ticker (e.g. volvo):").strip()

if query:
    try:
        matches = cached_search(query)
    except Exception as e:
        matches = []
        st.warning(f"Search failed: {e}")

    labels = {m["symbol"]: f"{m['name']} ({m['symbol']}) - {m['exchange']}" for m in matches}
    # Allow using the query as a ticker directly, e.g. if search misses it
    if " " not in query and query.upper() not in labels:
        labels[query.upper()] = f"{query.upper()} (use as ticker)"

    st.multiselect(
        "Search results",
        options=list(labels),
        format_func=lambda symbol: labels[symbol],
        key="search_pick",
    )
    st.button("Add to portfolio", on_click=add_selected_to_portfolio)

st.multiselect(
    "Portfolio",
    options=st.session_state.portfolio,
    key="portfolio_select",
    on_change=sync_portfolio,
)

run_button = st.button("Run Optimization")

if run_button:
    tickers = st.session_state.portfolio.copy()

    if not tickers:
        st.error("Please enter at least one ticker symbol.")
    else:
        st.session_state.analysis_done = True
        st.session_state.tickers = tickers

if st.session_state.get("analysis_done", False):
    tickers = st.session_state.tickers

    try:
        analysis = run_analysis(tuple(tickers), start, end, base_currency, risk_free_rate, max_weight)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    prices = analysis["prices"]
    expected_returns = analysis["expected_returns"]
    cov_matrix = analysis["cov_matrix"]
    opt_sharpe_weights = analysis["sharpe_weights"]
    opt_minvol_weights = analysis["minvol_weights"]

    st.caption(
        f"Data from {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, "
        f"all prices converted to {base_currency}."
    )

    # Max sharpe
    opt_sharpe_return, opt_sharpe_vol = calculate_portfolio_performance(opt_sharpe_weights, expected_returns, cov_matrix)

    # Min volatility
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

    fig = plot_efficient_frontier(
        analysis["mc_results"],
        analysis["frontier_volatilities"],
        analysis["frontier_returns"],
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
        manual_weights_raw[ticker] = st.slider(
            f"{ticker}", 0.0, 100.0, round(100 / len(tickers), 1),
            step=0.1, format="%.1f", key=f"weight_{ticker}"
        )

    raw_weights = np.array([manual_weights_raw[t] for t in tickers], dtype=float)

    backtest_weights = {"Max Sharpe": opt_sharpe_weights, "Min Volatility": opt_minvol_weights}

    if np.sum(raw_weights) == 0:
        st.error("At least one weight must be greater than zero")
    else:
        normalized_weights = raw_weights / np.sum(raw_weights)
        backtest_weights["Manual"] = normalized_weights

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

    st.subheader("Historical Performance")

    backtest = pd.DataFrame({
        name: calculate_buy_and_hold_value(prices, weights, INITIAL_VALUE)
        for name, weights in backtest_weights.items()
    })

    benchmark_symbol = BENCHMARKS[benchmark_name]
    if benchmark_symbol is not None:
        try:
            benchmark = load_benchmark(benchmark_symbol, start, end, base_currency)
            # Align the benchmark to the portfolio's trading days
            benchmark = benchmark.reindex(benchmark.index.union(backtest.index)).ffill().reindex(backtest.index)
            backtest[benchmark_name] = benchmark / benchmark.dropna().iloc[0] * INITIAL_VALUE
        except (ValueError, IndexError) as e:
            st.warning(f"Could not load benchmark {benchmark_name}: {e}")

    st.line_chart(backtest, x_label="Date", y_label=f"Value ({base_currency})")

    final_values = backtest.ffill().iloc[-1]
    st.dataframe(
        pd.DataFrame({
            "Final Value": final_values,
            "Total Return": (final_values / INITIAL_VALUE - 1) * 100,
        }),
        column_config={
            "Final Value": st.column_config.NumberColumn(format=f"%.0f {base_currency}"),
            "Total Return": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        f"{INITIAL_VALUE:,} {base_currency} invested on the first day and held without rebalancing. "
        "The optimized portfolios are calculated from this same period, so their results here "
        "are better than what could have been known in advance."
    )
