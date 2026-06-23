"""
Optional robustness tools for the research notebook.

Purpose
-------
These functions are meant to add a statistical check around the reported
Sharpe ratios. The idea is not to "prove" that a strategy works, but to show
how uncertain the historical Sharpe estimate is.

Recommended use in the final research notebook:
1. Put the numerical helper functions in module.py if you want the cleanest
   final structure.
2. Or paste this whole section into the research notebook as an optional
   robustness section.

The numerical calculations use NumPy. Matplotlib is only used for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt


def simple_returns_from_curve(curve):
    """
    Convert a wealth curve into simple one-period returns.
    """
    values = np.asarray(curve, dtype=float)
    return values[1:] / values[:-1] - 1.0


def annualized_sharpe_ratio(returns, periods_per_year=250):
    """
    Compute annualized Sharpe ratio with zero risk-free rate.

    This version avoids ready-made mean/std calls so the calculation stays
    transparent and close to the professor's NumPy-only expectation.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return np.nan

    mean_return = np.sum(returns) / len(returns)
    variance = np.sum((returns - mean_return) ** 2) / len(returns)
    volatility = np.sqrt(variance * periods_per_year)

    if volatility == 0.0:
        return np.nan

    return mean_return * periods_per_year / volatility


def make_block_bootstrap_indices(number_of_observations, block_length,
                                 random_generator):
    """
    Build one bootstrap sample by drawing consecutive return blocks.

    A block bootstrap is used because daily financial returns are not fully
    independent. Resampling blocks keeps some short-run time structure.
    """
    indices = []

    while len(indices) < number_of_observations:
        start_index = random_generator.integers(
            0,
            number_of_observations - block_length + 1,
        )

        block = np.arange(start_index, start_index + block_length)
        indices.extend(block)

    return np.asarray(indices[:number_of_observations], dtype=int)


def bootstrap_sharpe_distribution(returns, number_of_bootstraps=2000,
                                  block_length=20, seed=123):
    """
    Estimate the sampling distribution of one Sharpe ratio.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    random_generator = np.random.default_rng(seed)
    bootstrapped_sharpes = np.full(number_of_bootstraps, np.nan)

    for bootstrap_index in range(number_of_bootstraps):
        sample_indices = make_block_bootstrap_indices(
            number_of_observations=len(returns),
            block_length=block_length,
            random_generator=random_generator,
        )

        sample_returns = returns[sample_indices]
        bootstrapped_sharpes[bootstrap_index] = annualized_sharpe_ratio(
            sample_returns
        )

    return bootstrapped_sharpes


def bootstrap_sharpe_difference_distribution(strategy_returns,
                                             benchmark_returns,
                                             number_of_bootstraps=2000,
                                             block_length=20,
                                             seed=123):
    """
    Estimate uncertainty around:

        Sharpe(strategy) - Sharpe(benchmark)

    This is usually more useful than bootstrapping the strategy Sharpe alone,
    because it directly asks whether the strategy improvement is robust.
    """
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)

    number_of_observations = min(
        len(strategy_returns),
        len(benchmark_returns),
    )

    strategy_returns = strategy_returns[:number_of_observations]
    benchmark_returns = benchmark_returns[:number_of_observations]

    valid = (
        np.isfinite(strategy_returns)
        & np.isfinite(benchmark_returns)
    )

    strategy_returns = strategy_returns[valid]
    benchmark_returns = benchmark_returns[valid]

    random_generator = np.random.default_rng(seed)
    bootstrapped_differences = np.full(number_of_bootstraps, np.nan)

    for bootstrap_index in range(number_of_bootstraps):
        sample_indices = make_block_bootstrap_indices(
            number_of_observations=len(strategy_returns),
            block_length=block_length,
            random_generator=random_generator,
        )

        strategy_sample = strategy_returns[sample_indices]
        benchmark_sample = benchmark_returns[sample_indices]

        strategy_sharpe = annualized_sharpe_ratio(strategy_sample)
        benchmark_sharpe = annualized_sharpe_ratio(benchmark_sample)

        bootstrapped_differences[bootstrap_index] = (
            strategy_sharpe - benchmark_sharpe
        )

    return bootstrapped_differences


def confidence_interval(values, lower_percentile=2.5, upper_percentile=97.5):
    """
    Return a percentile confidence interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    lower = np.percentile(values, lower_percentile)
    upper = np.percentile(values, upper_percentile)

    return lower, upper


def summarize_bootstrap(values):
    """
    Return a compact summary dictionary for a bootstrap distribution.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    lower, upper = confidence_interval(values)

    return {
        "Bootstrap Mean": np.sum(values) / len(values),
        "CI Lower": lower,
        "CI Upper": upper,
        "Probability Above Zero": np.sum(values > 0.0) / len(values),
    }


def plot_bootstrap_distribution(values, title, xlabel,
                                reference_value=0.0):
    """
    Plot one bootstrap distribution with a vertical reference line.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    lower, upper = confidence_interval(values)

    plt.figure(figsize=(9, 4))
    plt.hist(values, bins=40, color="steelblue", alpha=0.75)
    plt.axvline(reference_value, color="black", linestyle="--",
                linewidth=1.2, label="Reference")
    plt.axvline(lower, color="darkorange", linestyle=":",
                linewidth=1.5, label="95% interval")
    plt.axvline(upper, color="darkorange", linestyle=":",
                linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Bootstrap frequency")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.show()


# ---------------------------------------------------------------------------
# Example usage for the research notebook
# ---------------------------------------------------------------------------
#
# Put this after one signal has already been evaluated out of sample.
# For example, after Signal 0 has produced:
# - strategy_0_out
# - buy_hold_0_out
#
# strategy_returns = simple_returns_from_curve(strategy_0_out)
# benchmark_returns = simple_returns_from_curve(buy_hold_0_out)
#
# sharpe_distribution = bootstrap_sharpe_distribution(
#     strategy_returns,
#     number_of_bootstraps=2000,
#     block_length=20,
#     seed=123,
# )
#
# sharpe_difference_distribution = bootstrap_sharpe_difference_distribution(
#     strategy_returns,
#     benchmark_returns,
#     number_of_bootstraps=2000,
#     block_length=20,
#     seed=123,
# )
#
# print("Bootstrap Sharpe interval:")
# print(confidence_interval(sharpe_distribution))
#
# print("Bootstrap Sharpe-difference summary:")
# print(summarize_bootstrap(sharpe_difference_distribution))
#
# plot_bootstrap_distribution(
#     sharpe_difference_distribution,
#     title="Signal 0: Bootstrap Sharpe Difference vs Buy-and-Hold",
#     xlabel="Sharpe(strategy) - Sharpe(buy-and-hold)",
#     reference_value=0.0,
# )
#
# Interpretation:
# If most of the distribution is above zero, the strategy's Sharpe improvement
# is more robust. If the interval crosses zero, the improvement exists in the
# point estimate but should be interpreted cautiously.
