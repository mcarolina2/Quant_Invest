import os
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR ="data"


@st.cache_data(ttl=3600)
def load_data(
    tickers,
    benchmark,
    start,
    end
):

    prices_dict = {}

    for ticker in tickers:

        file_path = os.path.join(
            DATA_DIR,
            f"{ticker.replace('^','')}.parquet"
        )

        if not os.path.exists(file_path):
            continue

        df = pd.read_parquet(file_path)

        df = df.loc[start:end]

        prices_dict[ticker] = df["Close"]

    if len(prices_dict) == 0:
        return None

    prices = pd.DataFrame(prices_dict)

    benchmark_path = os.path.join(
        DATA_DIR,
        f"{benchmark.replace('^','')}.parquet"
    )

    if os.path.exists(benchmark_path):

        benchmark_prices = (
            pd.read_parquet(benchmark_path)
            .loc[start:end][["Close"]]
        )

    else:

        benchmark_prices = pd.DataFrame()

    returns = np.log(
        prices / prices.shift(1)
    ).dropna()

    if not benchmark_prices.empty:

        mkt_returns = np.log(
            benchmark_prices /
            benchmark_prices.shift(1)
        ).dropna().squeeze()

        common_idx = returns.index.intersection(
            mkt_returns.index
        )

        returns = returns.loc[common_idx]
        mkt_returns = mkt_returns.loc[common_idx]

    else:

        mkt_returns = pd.Series(dtype=float)

    return {
        "prices": prices,
        "returns": returns,
        "benchmark_prices": benchmark_prices,
        "mkt_returns": mkt_returns,
    }