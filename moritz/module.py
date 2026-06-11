import numpy as np
import pandas as pd
from yahooquery import Ticker


def download_stock_price_data(tickers, start_date, end_date):
    raw = Ticker(tickers).history(start=start_date, end=end_date)
    close_prices = raw["close"].unstack(level=0)

    if "adjclose" in raw.columns:
        adjusted_prices = raw["adjclose"].unstack(level=0)
        df_prices = adjusted_prices.where(~adjusted_prices.isna(), close_prices)
    else:
        df_prices = close_prices

    df_prices = df_prices[tickers].dropna()
    df_prices.index = pd.to_datetime(df_prices.index)

    return df_prices, price_changes_from_price_frame(df_prices)


def price_changes_from_price_frame(df_prices):
    prices = df_prices.to_numpy(dtype=float)

    price_changes = np.ones(prices.shape)
    price_changes[1:, :] = prices[1:, :] / prices[:-1, :]

    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = price_changes

    return df_price_changes


def moving_average(price_series, window_length):
    prices = price_series.to_numpy(dtype=float)

    average = np.full(len(prices), np.nan)

    for index in range(window_length - 1, len(prices)):
        average[index] = np.sum(
            prices[index - window_length + 1:index + 1]
        ) / window_length

    return pd.Series(average, index=price_series.index)


def price_momentum(price_series, lookback_window):
    prices = price_series.to_numpy(dtype=float)

    momentum = np.full(len(prices), np.nan)
    momentum[lookback_window:] = (
        prices[lookback_window:] / prices[:-lookback_window] - 1.0
    )

    return pd.Series(momentum, index=price_series.index)


def portfolio_returns_from_values(portfolio_values):
    values = np.asarray(portfolio_values, dtype=float)
    return values[1:] / values[:-1] - 1.0


def annualized_return(portfolio_values, periods_per_year=250):
    values = np.asarray(portfolio_values, dtype=float)

    total_growth = values[-1] / values[0]
    number_of_periods = len(values) - 1

    if total_growth <= 0.0 or number_of_periods <= 0:
        return np.nan

    return total_growth ** (periods_per_year / number_of_periods) - 1.0


def annualized_volatility(returns, periods_per_year=250):
    returns = np.asarray(returns, dtype=float)

    mean_return = np.sum(returns) / len(returns)
    variance = np.sum((returns - mean_return) ** 2) / len(returns)

    return np.sqrt(variance * periods_per_year)


def sharpe_ratio(returns, periods_per_year=250):
    returns = np.asarray(returns, dtype=float)
    volatility = annualized_volatility(returns, periods_per_year)

    if volatility == 0.0:
        return np.nan

    mean_return = np.sum(returns) / len(returns)

    return mean_return * periods_per_year / volatility


def maximum_drawdown(portfolio_values):
    values = np.asarray(portfolio_values, dtype=float)

    running_maximum = np.maximum.accumulate(values)
    drawdowns = values / running_maximum - 1.0

    return np.min(drawdowns)


def performance_statistics_from_values(portfolio_values, periods_per_year=250):
    values = np.asarray(portfolio_values, dtype=float)
    returns = portfolio_returns_from_values(values)

    return {
        "Annualized Return": annualized_return(values, periods_per_year),
        "Annualized Volatility": annualized_volatility(returns, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(returns, periods_per_year),
        "Max Drawdown": maximum_drawdown(values)
    }


def signal_dataframe(index, raw_signal, **extra_columns):
    signal = np.asarray(raw_signal, dtype=float)
    signal = np.where(np.isfinite(signal), signal, 0.0)
    signal = np.where(signal > 0.0, 1.0, 0.0)

    position_change = np.zeros(len(signal))
    position_change[1:] = signal[1:] - signal[:-1]

    signal_frame = pd.DataFrame(index=index)
    signal_frame["signal"] = signal
    signal_frame["position_change"] = position_change

    for name, values in extra_columns.items():
        signal_frame[name] = values

    return signal_frame


def apply_execution_delay(signal_frame, delay_days=1):
    delayed_signal = (
        signal_frame["signal"]
        .shift(delay_days)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    delayed = signal_frame.copy(deep=True)
    delayed["signal"] = delayed_signal

    position_change = np.zeros(len(delayed_signal))
    position_change[1:] = delayed_signal[1:] - delayed_signal[:-1]

    delayed["position_change"] = position_change

    return delayed


def single_signal_portfolio_values(price_values, signal_values, initial_value=1.0):
    prices = np.asarray(price_values, dtype=float)
    signals = np.asarray(signal_values, dtype=float)

    portfolio_values = np.ones(len(prices)) * initial_value

    for index in range(1, len(prices)):
        asset_return = prices[index] / prices[index - 1] - 1.0
        strategy_return = signals[index - 1] * asset_return
        portfolio_values[index] = portfolio_values[index - 1] * (1.0 + strategy_return)

    return portfolio_values


def vix_term_structure_signal(
    target_series,
    vix_series,
    vix3m_series,
    ratio_threshold,
    momentum_window,
    momentum_threshold
):
    target_series = target_series.dropna()

    vix = vix_series.reindex(target_series.index).to_numpy(dtype=float)
    vix3m = vix3m_series.reindex(target_series.index).to_numpy(dtype=float)

    vix_ratio = vix3m / vix
    target_momentum = price_momentum(target_series, momentum_window).to_numpy(dtype=float)

    signal = np.where(
        (vix_ratio > ratio_threshold)
        & (target_momentum > momentum_threshold),
        1.0,
        0.0
    )

    signal[:momentum_window] = 0.0

    return signal_dataframe(
        target_series.index,
        signal,
        vix_ratio=vix_ratio,
        target_momentum=target_momentum
    )


def oil_energy_relative_strength_signal(
    target_series,
    oil_series,
    market_series,
    trend_window,
    oil_momentum_window,
    oil_momentum_threshold,
    target_momentum_window,
    target_momentum_threshold,
    relative_strength_window,
    relative_strength_threshold
):
    target_series = target_series.dropna()

    target_prices = target_series.to_numpy(dtype=float)
    oil = oil_series.reindex(target_series.index)
    market = market_series.reindex(target_series.index)

    trend_average = moving_average(target_series, trend_window).to_numpy(dtype=float)

    oil_momentum = price_momentum(
        oil,
        oil_momentum_window
    ).to_numpy(dtype=float)

    target_momentum = price_momentum(
        target_series,
        target_momentum_window
    ).to_numpy(dtype=float)

    relative_strength = target_series / market

    relative_strength_momentum = price_momentum(
        relative_strength,
        relative_strength_window
    ).to_numpy(dtype=float)

    signal = np.where(
        (target_prices > trend_average)
        & (oil_momentum > oil_momentum_threshold)
        & (target_momentum > target_momentum_threshold)
        & (relative_strength_momentum > relative_strength_threshold),
        1.0,
        0.0
    )

    warmup_period = max(
        trend_window,
        oil_momentum_window,
        target_momentum_window,
        relative_strength_window
    )

    signal[:warmup_period] = 0.0

    return signal_dataframe(
        target_series.index,
        signal,
        trend_average=trend_average,
        oil_momentum=oil_momentum,
        target_momentum=target_momentum,
        relative_strength_momentum=relative_strength_momentum
    )


def gold_regime_signal(
    gold_series,
    bond_series,
    trend_window,
    gold_momentum_window,
    gold_momentum_threshold,
    bond_momentum_window,
    bond_momentum_threshold
):
    gold_series = gold_series.dropna()

    gold_prices = gold_series.to_numpy(dtype=float)
    bonds = bond_series.reindex(gold_series.index)

    trend_average = moving_average(
        gold_series,
        trend_window
    ).to_numpy(dtype=float)

    gold_momentum = price_momentum(
        gold_series,
        gold_momentum_window
    ).to_numpy(dtype=float)

    bond_momentum = price_momentum(
        bonds,
        bond_momentum_window
    ).to_numpy(dtype=float)

    signal = np.where(
        (gold_prices > trend_average)
        & (gold_momentum > gold_momentum_threshold)
        & (bond_momentum > bond_momentum_threshold),
        1.0,
        0.0
    )

    warmup_period = max(
        trend_window,
        gold_momentum_window,
        bond_momentum_window
    )

    signal[:warmup_period] = 0.0

    return signal_dataframe(
        gold_series.index,
        signal,
        trend_average=trend_average,
        gold_momentum=gold_momentum,
        bond_momentum=bond_momentum
    )


def ita_defensive_volatility_signal(
    ita_series,
    spy_series,
    vix_series,
    xlu_series,
    xlp_series,
    trend_window,
    ita_momentum_window,
    ita_momentum_threshold,
    relative_strength_window,
    relative_strength_threshold,
    vix_average_window,
    defensive_momentum_window,
    defensive_momentum_threshold
):
    ita_prices = ita_series.to_numpy(dtype=float)

    spy = spy_series.reindex(ita_series.index)
    vix = vix_series.reindex(ita_series.index)
    xlu = xlu_series.reindex(ita_series.index)
    xlp = xlp_series.reindex(ita_series.index)

    trend_average = moving_average(
        ita_series,
        trend_window
    ).to_numpy(dtype=float)

    ita_momentum = price_momentum(
        ita_series,
        ita_momentum_window
    ).to_numpy(dtype=float)

    relative_strength = ita_series / spy

    relative_momentum = price_momentum(
        relative_strength,
        relative_strength_window
    ).to_numpy(dtype=float)

    vix_average = moving_average(
        vix,
        vix_average_window
    ).to_numpy(dtype=float)

    defensive_ratio = ((xlu + xlp) / 2.0) / spy

    defensive_momentum = price_momentum(
        defensive_ratio,
        defensive_momentum_window
    ).to_numpy(dtype=float)

    vix_elevated = vix.to_numpy(dtype=float) > vix_average

    signal = np.where(
        (ita_prices > trend_average)
        & (ita_momentum > ita_momentum_threshold)
        & (relative_momentum > relative_strength_threshold)
        & (
            vix_elevated
            | (defensive_momentum > defensive_momentum_threshold)
        ),
        1.0,
        0.0
    )

    warmup_period = max(
        trend_window,
        ita_momentum_window,
        relative_strength_window,
        vix_average_window,
        defensive_momentum_window
    )

    signal[:warmup_period] = 0.0

    return signal_dataframe(
        ita_series.index,
        signal,
        trend_average=trend_average,
        ita_momentum=ita_momentum,
        relative_momentum=relative_momentum,
        vix_average=vix_average,
        defensive_momentum=defensive_momentum
    )
