import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -----------------------------
# Monte Carlo Simulation for XRP
# -----------------------------

TICKER = "XRP-USD"
LOOKBACK_PERIOD = "2y"
SIM_DAYS = 365
N_SIMULATIONS = 10_000
RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No price data returned. Check ticker or internet connection.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker]
    else:
        close = data["Close"]

    cleaned = pd.DataFrame({"close": close})
    return cleaned.dropna()


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["log_return"] = np.log(data["close"] / data["close"].shift(1))
    return data.dropna()


def simulate_gbm_paths(
    start_price: float,
    daily_mu: float,
    daily_sigma: float,
    sim_days: int,
    n_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Geometric Brownian Motion:

    S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma*Z_t)

    This is a simplified stochastic model, not a prediction engine.
    """

    if seed is not None:
        np.random.seed(seed)

    random_shocks = np.random.normal(
        loc=0,
        scale=1,
        size=(sim_days, n_simulations)
    )

    daily_growth = np.exp(
        (daily_mu - 0.5 * daily_sigma**2) + daily_sigma * random_shocks
    )

    paths = np.zeros((sim_days + 1, n_simulations))
    paths[0] = start_price
    paths[1:] = start_price * np.cumprod(daily_growth, axis=0)

    return paths


def summarize_simulation(paths: np.ndarray) -> pd.DataFrame:
    final_prices = paths[-1]

    summary = {
        "current_price": paths[0, 0],
        "mean_final_price": np.mean(final_prices),
        "median_final_price": np.median(final_prices),
        "5th_percentile": np.percentile(final_prices, 5),
        "25th_percentile": np.percentile(final_prices, 25),
        "75th_percentile": np.percentile(final_prices, 75),
        "95th_percentile": np.percentile(final_prices, 95),
        "probability_gain": np.mean(final_prices > paths[0, 0]),
        "probability_50pct_drawdown": np.mean(final_prices < paths[0, 0] * 0.5),
        "probability_100pct_gain": np.mean(final_prices > paths[0, 0] * 2),
    }

    return pd.DataFrame([summary])


def plot_sample_paths(paths: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(12, 6))

    sample_count = min(100, paths.shape[1])

    for i in range(sample_count):
        plt.plot(paths[:, i], linewidth=0.8, alpha=0.35)

    plt.title("Monte Carlo Simulation: XRP Potential Price Paths")
    plt.xlabel("Days Into Future")
    plt.ylabel("Simulated XRP Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_distribution(paths: np.ndarray, output_path: str) -> None:
    final_prices = paths[-1]

    plt.figure(figsize=(12, 6))
    plt.hist(final_prices, bins=80)
    plt.title("Distribution of Simulated XRP Prices After 1 Year")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    print("Fetching XRP data...")
    data = fetch_price_data(TICKER, LOOKBACK_PERIOD)
    data = calculate_returns(data)

    #debugging for data["close"].iloc[-1]
    print(type(data))
    print(data.head())
    print(data.columns)

    print(type(data["close"]))
    print(data["close"].iloc[-1])
    print(type(data["close"].iloc[-1]))

    start_price = float(data["close"].iloc[-1])
    daily_mu = float(data["log_return"].mean())
    daily_sigma = float(data["log_return"].std())

    annualized_return = daily_mu * 365
    annualized_volatility = daily_sigma * np.sqrt(365)

    print(f"\nLatest XRP price: ${start_price:.4f}")
    print(f"Estimated annualized return: {annualized_return:.2%}")
    print(f"Estimated annualized volatility: {annualized_volatility:.2%}")

    paths = simulate_gbm_paths(
        start_price=start_price,
        daily_mu=daily_mu,
        daily_sigma=daily_sigma,
        sim_days=SIM_DAYS,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    summary = summarize_simulation(paths)

    summary_path = os.path.join(OUTPUT_DIR, "xrp_simulation_summary.csv")
    paths_chart_path = os.path.join(OUTPUT_DIR, "xrp_simulated_paths.png")
    dist_chart_path = os.path.join(OUTPUT_DIR, "xrp_final_price_distribution.png")

    summary.to_csv(summary_path, index=False)

    print("\nSimulation Summary:")
    print(summary.T)

    plot_sample_paths(paths, paths_chart_path)
    plot_distribution(paths, dist_chart_path)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved path chart to: {paths_chart_path}")
    print(f"Saved distribution chart to: {dist_chart_path}")


if __name__ == "__main__":
    main()