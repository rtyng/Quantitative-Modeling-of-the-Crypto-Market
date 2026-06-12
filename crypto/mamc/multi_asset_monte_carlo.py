"""
Multi-Asset Monte Carlo Simulation for Crypto Price Risk

Project:
    quant/crypto

Purpose:
    Simulate one-year potential price paths for multiple crypto assets using a
    correlated Geometric Brownian Motion model.

What this script does:
    1. Downloads historical daily price data from Yahoo Finance.
    2. Calculates daily log returns for each crypto asset.
    3. Estimates:
        - daily mean returns
        - daily volatility
        - correlation matrix
        - covariance matrix
    4. Uses Cholesky decomposition to generate correlated random shocks.
    5. Simulates future price paths for XRP, BTC, ETH, SOL, and other selected assets.
    6. Saves summary statistics and charts to the outputs folder.

Important modeling note:
    This is not a prediction engine. It is a probabilistic risk model based on
    historical return behavior. Crypto markets are highly regime-dependent and
    can be strongly affected by legal, regulatory, liquidity, macroeconomic,
    exchange, and Bitcoin-led market events.

Professional interpretation:
    The model is useful for studying possible distributions of outcomes,
    downside risk, upside tails, asset correlations, and portfolio behavior.
    It should not be interpreted as "where the price will go."

Model limitations:
    - Assumes historical drift and volatility are useful estimates.
    - Assumes log returns are approximately normal.
    - Assumes correlations remain stable.
    - Does not yet include event shocks such as Senate CLARITY Act outcomes.
    - Does not model volatility clustering.
    - Does not model liquidity, order books, or on-chain flows.

Future upgrades:
    - Add regulatory scenario shocks.
    - Add BTC regime filters.
    - Add GARCH-style volatility.
    - Add portfolio allocation weights.
    - Add Value-at-Risk and Expected Shortfall.
    - Add rolling correlation analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -----------------------------
# User Settings
# -----------------------------

TICKERS = [
    "XRP-USD",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "XLM-USD",
    "LINK-USD",
    "LTC-USD",
    "DOGE-USD",
]

LOOKBACK_PERIOD = "2y"
SIM_DAYS = 365
N_SIMULATIONS = 10_000
RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Data Collection
# -----------------------------

def fetch_multi_asset_prices(tickers: list[str], period: str) -> pd.DataFrame:
    """
    Download adjusted daily close prices for several crypto assets.

    Returns:
        A dataframe shaped like:

            Date        XRP-USD   BTC-USD   ETH-USD ...
            2024-...    0.52     69000.0   3500.0
    """

    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data returned. Check internet connection or tickers.")

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"].copy()
    else:
        close_prices = data[["Close"]].copy()
        close_prices.columns = tickers

    close_prices = close_prices.dropna(how="any")

    return close_prices


def calculate_log_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily log returns.

    Formula:
        log_return_t = ln(price_t / price_t-1)
    """

    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna(how="any")


# -----------------------------
# Monte Carlo Engine
# -----------------------------

def simulate_correlated_gbm(
    start_prices: pd.Series,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    sim_days: int,
    n_simulations: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Simulate correlated Geometric Brownian Motion paths.

    This is the core model.

    For each asset:

        S_t = S_0 * exp((mu - 0.5 * sigma^2) + shock)

    But unlike the single-asset model, the random shocks are correlated across
    assets using the covariance matrix and Cholesky decomposition.

    Returns:
        Dictionary where each key is a ticker and each value is an array with
        shape:

            rows    = days
            columns = simulations
    """

    if seed is not None:
        np.random.seed(seed)

    tickers = list(start_prices.index)
    n_assets = len(tickers)

    cov = covariance_matrix.values

    try:
        cholesky_matrix = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix is not positive definite. "
            "Try reducing the ticker list or increasing lookback period."
        )

    paths = {
        ticker: np.zeros((sim_days + 1, n_simulations))
        for ticker in tickers
    }

    for ticker in tickers:
        paths[ticker][0, :] = start_prices[ticker]

    for day in range(1, sim_days + 1):
        independent_randoms = np.random.normal(size=(n_assets, n_simulations))
        correlated_shocks = cholesky_matrix @ independent_randoms

        for i, ticker in enumerate(tickers):
            mu = mean_returns[ticker]
            sigma_squared = covariance_matrix.loc[ticker, ticker]

            growth_factor = np.exp(
                (mu - 0.5 * sigma_squared) + correlated_shocks[i]
            )

            paths[ticker][day, :] = paths[ticker][day - 1, :] * growth_factor

    return paths


# -----------------------------
# Summary Statistics
# -----------------------------

def summarize_asset_paths(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Generate summary statistics for each simulated asset.
    """

    rows = []

    for ticker, asset_paths in paths.items():
        start_price = asset_paths[0, 0]
        final_prices = asset_paths[-1, :]

        row = {
            "ticker": ticker,
            "current_price": start_price,
            "mean_final_price": np.mean(final_prices),
            "median_final_price": np.median(final_prices),
            "5th_percentile": np.percentile(final_prices, 5),
            "25th_percentile": np.percentile(final_prices, 25),
            "75th_percentile": np.percentile(final_prices, 75),
            "95th_percentile": np.percentile(final_prices, 95),
            "probability_gain": np.mean(final_prices > start_price),
            "probability_50pct_drawdown": np.mean(final_prices < start_price * 0.5),
            "probability_100pct_gain": np.mean(final_prices > start_price * 2),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def plot_asset_paths(
    paths: dict[str, np.ndarray],
    ticker: str,
    output_path: str,
    sample_count: int = 100,
) -> None:
    """
    Plot sample simulated paths for one asset.
    """

    asset_paths = paths[ticker]
    sample_count = min(sample_count, asset_paths.shape[1])

    plt.figure(figsize=(12, 6))

    for i in range(sample_count):
        plt.plot(asset_paths[:, i], linewidth=0.8, alpha=0.35)

    plt.title(f"Monte Carlo Simulation: {ticker} Potential Price Paths")
    plt.xlabel("Days Into Future")
    plt.ylabel(f"Simulated {ticker} Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_final_price_distribution(
    paths: dict[str, np.ndarray],
    ticker: str,
    output_path: str,
) -> None:
    """
    Plot final simulated price distribution for one asset.
    """

    final_prices = paths[ticker][-1, :]

    plt.figure(figsize=(12, 6))
    plt.hist(final_prices, bins=80)
    plt.title(f"Distribution of Simulated {ticker} Prices After 1 Year")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_correlation_matrix(correlation_matrix: pd.DataFrame, output_path: str) -> None:
    """
    Plot crypto asset return correlation matrix.
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, aspect="auto")
    plt.colorbar(label="Correlation")

    plt.xticks(
        ticks=range(len(correlation_matrix.columns)),
        labels=correlation_matrix.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=range(len(correlation_matrix.index)),
        labels=correlation_matrix.index,
    )

    plt.title("Historical Daily Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


# -----------------------------
# Main Program
# -----------------------------

def main() -> None:
    print("Fetching crypto price data...")

    price_data = fetch_multi_asset_prices(TICKERS, LOOKBACK_PERIOD)
    log_returns = calculate_log_returns(price_data)

    start_prices = price_data.iloc[-1]
    mean_returns = log_returns.mean()
    covariance_matrix = log_returns.cov()
    correlation_matrix = log_returns.corr()

    print("\nLatest Prices:")
    print(start_prices)

    print("\nDaily Mean Returns:")
    print(mean_returns)

    print("\nAnnualized Volatility:")
    annualized_volatility = log_returns.std() * np.sqrt(365)
    print(annualized_volatility)

    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    paths = simulate_correlated_gbm(
        start_prices=start_prices,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        sim_days=SIM_DAYS,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    summary = summarize_asset_paths(paths)

    summary_path = os.path.join(OUTPUT_DIR, "multi_asset_simulation_summary.csv")
    correlation_path = os.path.join(OUTPUT_DIR, "crypto_correlation_matrix.csv")
    correlation_chart_path = os.path.join(OUTPUT_DIR, "crypto_correlation_matrix.png")

    summary.to_csv(summary_path, index=False)
    correlation_matrix.to_csv(correlation_path)

    print("\nSimulation Summary:")
    print(summary)

    plot_correlation_matrix(correlation_matrix, correlation_chart_path)

    for ticker in TICKERS:
        safe_name = ticker.replace("-", "_")

        paths_chart_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_simulated_paths.png"
        )

        distribution_chart_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_final_price_distribution.png"
        )

        plot_asset_paths(paths, ticker, paths_chart_path)
        plot_final_price_distribution(paths, ticker, distribution_chart_path)

    print("\nSaved files:")
    print(f"Summary: {summary_path}")
    print(f"Correlation CSV: {correlation_path}")
    print(f"Correlation chart: {correlation_chart_path}")
    print(f"Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()