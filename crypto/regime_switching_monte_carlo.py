"""
Regime-Switching Multi-Asset Monte Carlo Simulation for Crypto

Project:
    quant/crypto

Purpose:
    Simulate future crypto prices using four market regimes:

        1. Bull Market Regime
        2. Neutral Market Regime
        3. Bear Market Regime
        4. Regulatory Shock Regime

Core idea:
    Instead of assuming one stable return/volatility/correlation structure,
    this model assumes crypto moves through different regimes over time.

    Each regime has:
        - different expected return behavior
        - different volatility
        - different correlation intensity
        - different probability of transitioning to another regime

Included variable categories:
    Market variables:
        - BTC, ETH, SOL, XRP, ADA, XLM, LINK, DOGE, LTC

    Macro variables:
        - 10-year Treasury yield proxy: ^TNX
        - U.S. Dollar Index proxy: DX-Y.NYB
        - Nasdaq proxy: QQQ

    Regulatory/event variables:
        - Manual regulatory shock assumptions
        - Used to model CLARITY Act-style outcomes

    On-chain variables:
        - Placeholder hooks included
        - This script does not yet fetch live on-chain data

    Derivatives variables:
        - Placeholder hooks included
        - This script does not yet fetch funding/open interest data

Important:
    This is a research/risk simulation tool, not an investment prediction engine.

Outputs:
    All files are saved to:

        outputs/regime_switching_monte_carlo/

Files created:
    - regime_simulation_summary.csv
    - regime_counts.csv
    - regime_path_example.csv
    - correlation_matrix.csv
    - simulated price charts
    - final price distribution charts
    - regime path chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# User Settings
# ============================================================

CRYPTO_TICKERS = [
    "XRP-USD",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "XLM-USD",
    "LINK-USD",
    "DOGE-USD",
    "LTC-USD",
]

MACRO_TICKERS = [
    "^TNX",       # 10-year Treasury yield proxy
    "DX-Y.NYB",   # U.S. Dollar Index proxy
    "QQQ",        # Nasdaq / risk appetite proxy
]

LOOKBACK_PERIOD = "3y"
SIM_DAYS = 365
N_SIMULATIONS = 10_000
RANDOM_SEED = 42

BASE_OUTPUT_DIR = "outputs"
PROJECT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "regime_switching_monte_carlo")
os.makedirs(PROJECT_OUTPUT_DIR, exist_ok=True)


# ============================================================
# Regime Definitions
# ============================================================

REGIMES = {
    0: "bull",
    1: "neutral",
    2: "bear",
    3: "regulatory_shock",
}

REGIME_IDS = list(REGIMES.keys())


"""
Transition matrix.

Rows = current regime
Columns = next regime

Example:
    If today is bull:
        85% chance tomorrow is still bull
        10% chance tomorrow is neutral
        4% chance tomorrow is bear
        1% chance tomorrow is regulatory shock
"""

TRANSITION_MATRIX = np.array([
    [0.85, 0.10, 0.04, 0.01],  # bull
    [0.15, 0.70, 0.10, 0.05],  # neutral
    [0.05, 0.20, 0.70, 0.05],  # bear
    [0.20, 0.35, 0.25, 0.20],  # regulatory shock
])


"""
Regime multipliers.

These modify the historical estimates.

drift_multiplier:
    Changes expected return.

volatility_multiplier:
    Changes daily volatility.

correlation_blend:
    Pushes correlations closer to 1 during stress regimes.
    This models correlation convergence during crashes.

event_shock_mean:
    One-day shock applied during regulatory shock days.

event_shock_std:
    Shock uncertainty during regulatory shock days.
"""

REGIME_PARAMS = {
    "bull": {
        "drift_multiplier": 1.75,
        "volatility_multiplier": 0.90,
        "correlation_blend": 0.10,
        "event_shock_mean": 0.00,
        "event_shock_std": 0.00,
    },
    "neutral": {
        "drift_multiplier": 0.60,
        "volatility_multiplier": 1.00,
        "correlation_blend": 0.00,
        "event_shock_mean": 0.00,
        "event_shock_std": 0.00,
    },
    "bear": {
        "drift_multiplier": -1.25,
        "volatility_multiplier": 1.60,
        "correlation_blend": 0.25,
        "event_shock_mean": 0.00,
        "event_shock_std": 0.00,
    },
    "regulatory_shock": {
        "drift_multiplier": 0.00,
        "volatility_multiplier": 2.25,
        "correlation_blend": 0.35,
        "event_shock_mean": -0.025,
        "event_shock_std": 0.08,
    },
}


"""
Asset-specific regulatory shock sensitivity.

Higher value means the asset reacts more strongly to regulatory shock days.

XRP and XLM are given higher sensitivity because their narratives are more
payments/regulatory/banking-adjacent.
"""

REGULATORY_SENSITIVITY = {
    "XRP-USD": 1.50,
    "XLM-USD": 1.25,
    "BTC-USD": 0.75,
    "ETH-USD": 0.90,
    "SOL-USD": 1.10,
    "ADA-USD": 1.00,
    "LINK-USD": 0.90,
    "DOGE-USD": 1.15,
    "LTC-USD": 0.80,
}


# ============================================================
# Data Functions
# ============================================================

def fetch_close_prices(tickers: list[str], period: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers

    return close.dropna(how="any")


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="any")


# ============================================================
# Regime Logic
# ============================================================

def infer_starting_regime(
    crypto_returns: pd.DataFrame,
    macro_returns: pd.DataFrame | None = None,
) -> int:
    """
    Simple rule-based regime classifier.

    This decides the starting regime based on recent BTC behavior.

    Professional version would use:
        - Hidden Markov Models
        - macro liquidity data
        - volatility regimes
        - trend filters
        - funding/open interest
        - on-chain exchange flows
    """

    btc_recent = crypto_returns["BTC-USD"].tail(30)
    btc_90 = crypto_returns["BTC-USD"].tail(90)

    recent_return = btc_recent.sum()
    medium_return = btc_90.sum()
    recent_vol = btc_recent.std() * np.sqrt(365)

    if recent_return > 0.15 and medium_return > 0.20:
        return 0  # bull

    if recent_return < -0.15 or recent_vol > 0.95:
        return 2  # bear

    return 1  # neutral


def simulate_regime_paths(
    n_days: int,
    n_simulations: int,
    starting_regime: int,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    regime_paths = np.zeros((n_days + 1, n_simulations), dtype=int)
    regime_paths[0, :] = starting_regime

    for sim in range(n_simulations):
        for day in range(1, n_days + 1):
            current_regime = regime_paths[day - 1, sim]
            regime_paths[day, sim] = np.random.choice(
                REGIME_IDS,
                p=transition_matrix[current_regime],
            )

    return regime_paths


def blend_correlation_matrix(correlation: pd.DataFrame, blend_strength: float) -> pd.DataFrame:
    """
    Push correlations toward 1.0.

    This approximates crisis behavior where crypto assets increasingly move
    together during market stress.
    """

    corr = correlation.copy()
    ones = pd.DataFrame(
        np.ones(corr.shape),
        index=corr.index,
        columns=corr.columns,
    )

    blended = (1 - blend_strength) * corr + blend_strength * ones
    np.fill_diagonal(blended.values, 1.0)

    return blended


def covariance_from_correlation(
    correlation: pd.DataFrame,
    volatilities: pd.Series,
) -> pd.DataFrame:
    diagonal_vol = np.diag(volatilities.values)
    cov_values = diagonal_vol @ correlation.values @ diagonal_vol

    return pd.DataFrame(
        cov_values,
        index=correlation.index,
        columns=correlation.columns,
    )


# ============================================================
# Monte Carlo Engine
# ============================================================

def simulate_regime_switching_prices(
    start_prices: pd.Series,
    base_mean_returns: pd.Series,
    base_covariance: pd.DataFrame,
    base_correlation: pd.DataFrame,
    regime_paths: np.ndarray,
) -> dict[str, np.ndarray]:
    tickers = list(start_prices.index)
    n_assets = len(tickers)
    n_days = regime_paths.shape[0] - 1
    n_simulations = regime_paths.shape[1]

    base_daily_vol = pd.Series(
        np.sqrt(np.diag(base_covariance.values)),
        index=tickers,
    )

    paths = {
        ticker: np.zeros((n_days + 1, n_simulations))
        for ticker in tickers
    }

    for ticker in tickers:
        paths[ticker][0, :] = start_prices[ticker]

    for sim in range(n_simulations):
        for day in range(1, n_days + 1):
            regime_id = regime_paths[day, sim]
            regime_name = REGIMES[regime_id]
            params = REGIME_PARAMS[regime_name]

            adjusted_mean = base_mean_returns * params["drift_multiplier"]
            adjusted_vol = base_daily_vol * params["volatility_multiplier"]

            adjusted_corr = blend_correlation_matrix(
                base_correlation,
                params["correlation_blend"],
            )

            adjusted_cov = covariance_from_correlation(
                adjusted_corr,
                adjusted_vol,
            )

            try:
                cholesky = np.linalg.cholesky(adjusted_cov.values)
            except np.linalg.LinAlgError:
                adjusted_cov = adjusted_cov + np.eye(n_assets) * 1e-10
                cholesky = np.linalg.cholesky(adjusted_cov.values)

            independent_randoms = np.random.normal(size=n_assets)
            correlated_shocks = cholesky @ independent_randoms

            event_shock = 0.0

            if regime_name == "regulatory_shock":
                event_shock = np.random.normal(
                    params["event_shock_mean"],
                    params["event_shock_std"],
                )

            for i, ticker in enumerate(tickers):
                sigma_squared = adjusted_cov.loc[ticker, ticker]

                sensitivity = REGULATORY_SENSITIVITY.get(ticker, 1.0)
                ticker_event_shock = event_shock * sensitivity

                growth_factor = np.exp(
                    adjusted_mean[ticker]
                    - 0.5 * sigma_squared
                    + correlated_shocks[i]
                    + ticker_event_shock
                )

                paths[ticker][day, sim] = paths[ticker][day - 1, sim] * growth_factor

    return paths


# ============================================================
# Output / Analytics
# ============================================================

def summarize_paths(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    for ticker, asset_paths in paths.items():
        start_price = asset_paths[0, 0]
        final_prices = asset_paths[-1, :]

        rows.append({
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
            "max_simulated_final_price": np.max(final_prices),
            "min_simulated_final_price": np.min(final_prices),
        })

    return pd.DataFrame(rows)


def summarize_regimes(regime_paths: np.ndarray) -> pd.DataFrame:
    rows = []

    for regime_id, regime_name in REGIMES.items():
        count = np.sum(regime_paths == regime_id)
        percent = count / regime_paths.size

        rows.append({
            "regime_id": regime_id,
            "regime_name": regime_name,
            "count": count,
            "percent_of_simulated_days": percent,
        })

    return pd.DataFrame(rows)


def plot_asset_paths(paths: dict[str, np.ndarray], ticker: str, output_path: str) -> None:
    asset_paths = paths[ticker]
    sample_count = min(100, asset_paths.shape[1])

    plt.figure(figsize=(12, 6))

    for i in range(sample_count):
        plt.plot(asset_paths[:, i], linewidth=0.8, alpha=0.35)

    plt.title(f"Regime-Switching Monte Carlo: {ticker}")
    plt.xlabel("Days Into Future")
    plt.ylabel("Simulated Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_final_distribution(paths: dict[str, np.ndarray], ticker: str, output_path: str) -> None:
    final_prices = paths[ticker][-1, :]

    plt.figure(figsize=(12, 6))
    plt.hist(final_prices, bins=80)
    plt.title(f"Final Price Distribution After 1 Year: {ticker}")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regime_example(regime_paths: np.ndarray, output_path: str) -> None:
    example_path = regime_paths[:, 0]

    plt.figure(figsize=(12, 4))
    plt.step(range(len(example_path)), example_path, where="post")
    plt.yticks(list(REGIMES.keys()), list(REGIMES.values()))
    plt.title("Example Simulated Regime Path")
    plt.xlabel("Days Into Future")
    plt.ylabel("Regime")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_matrix(correlation: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, aspect="auto")
    plt.colorbar(label="Correlation")

    plt.xticks(
        ticks=range(len(correlation.columns)),
        labels=correlation.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=range(len(correlation.index)),
        labels=correlation.index,
    )

    plt.title("Base Historical Daily Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("Fetching crypto prices...")
    crypto_prices = fetch_close_prices(CRYPTO_TICKERS, LOOKBACK_PERIOD)
    crypto_returns = calculate_log_returns(crypto_prices)

    print("Fetching macro proxies...")
    try:
        macro_prices = fetch_close_prices(MACRO_TICKERS, LOOKBACK_PERIOD)
        macro_returns = calculate_log_returns(macro_prices)
    except Exception as error:
        print(f"Macro data fetch failed. Continuing without macro returns. Error: {error}")
        macro_returns = None

    start_prices = crypto_prices.iloc[-1]
    base_mean_returns = crypto_returns.mean()
    base_covariance = crypto_returns.cov()
    base_correlation = crypto_returns.corr()

    starting_regime = infer_starting_regime(crypto_returns, macro_returns)

    print("\nStarting regime:")
    print(REGIMES[starting_regime])

    print("\nLatest crypto prices:")
    print(start_prices)

    print("\nAnnualized crypto volatility:")
    print(crypto_returns.std() * np.sqrt(365))

    print("\nBase crypto correlation matrix:")
    print(base_correlation)

    print("\nSimulating regime paths...")
    regime_paths = simulate_regime_paths(
        n_days=SIM_DAYS,
        n_simulations=N_SIMULATIONS,
        starting_regime=starting_regime,
        transition_matrix=TRANSITION_MATRIX,
    )

    print("Simulating regime-switching price paths...")
    paths = simulate_regime_switching_prices(
        start_prices=start_prices,
        base_mean_returns=base_mean_returns,
        base_covariance=base_covariance,
        base_correlation=base_correlation,
        regime_paths=regime_paths,
    )

    summary = summarize_paths(paths)
    regime_summary = summarize_regimes(regime_paths)

    summary_path = os.path.join(PROJECT_OUTPUT_DIR, "regime_simulation_summary.csv")
    regime_counts_path = os.path.join(PROJECT_OUTPUT_DIR, "regime_counts.csv")
    regime_path_example_path = os.path.join(PROJECT_OUTPUT_DIR, "regime_path_example.csv")
    correlation_path = os.path.join(PROJECT_OUTPUT_DIR, "correlation_matrix.csv")

    summary.to_csv(summary_path, index=False)
    regime_summary.to_csv(regime_counts_path, index=False)
    base_correlation.to_csv(correlation_path)

    pd.DataFrame({
        "day": range(SIM_DAYS + 1),
        "regime_id": regime_paths[:, 0],
        "regime_name": [REGIMES[x] for x in regime_paths[:, 0]],
    }).to_csv(regime_path_example_path, index=False)

    plot_regime_example(
        regime_paths,
        os.path.join(PROJECT_OUTPUT_DIR, "regime_path_example.png"),
    )

    plot_correlation_matrix(
        base_correlation,
        os.path.join(PROJECT_OUTPUT_DIR, "base_correlation_matrix.png"),
    )

    for ticker in CRYPTO_TICKERS:
        safe_name = ticker.replace("-", "_")

        plot_asset_paths(
            paths,
            ticker,
            os.path.join(PROJECT_OUTPUT_DIR, f"{safe_name}_regime_paths.png"),
        )

        plot_final_distribution(
            paths,
            ticker,
            os.path.join(PROJECT_OUTPUT_DIR, f"{safe_name}_final_distribution.png"),
        )

    print("\nSimulation complete.")
    print(f"Outputs saved to: {PROJECT_OUTPUT_DIR}")

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()