import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def _build_proxy_factors(returns: pd.DataFrame, mkt_returns: pd.Series, rf_daily: float) -> tuple:
    """
    Build SMB and HML proxy factors from the asset universe.

    SMB proxy: average return of 'small' assets minus average return of 'large' assets
               (ranked by trailing annual return as a volatility proxy for size).

    HML proxy: average return of 'value' assets minus average return of 'growth' assets
               (ranked by inverse of prior-period return as a book-to-market proxy).
    """
    n = len(returns.columns)
    if n < 3:
        return None, None

    # Use cumulative return as a size proxy (high cum_ret ≈ large cap in this proxy)
    cum_ret = returns.mean()
    sorted_tickers = cum_ret.sort_values().index.tolist()
    split = max(1, n // 3)

    small_tickers = sorted_tickers[:split]
    large_tickers = sorted_tickers[-split:]

    # HML proxy: low prior return = "value", high prior return = "growth"
    value_tickers = sorted_tickers[:split]
    growth_tickers = sorted_tickers[-split:]

    smb = returns[small_tickers].mean(axis=1) - returns[large_tickers].mean(axis=1)
    hml = returns[value_tickers].mean(axis=1) - returns[growth_tickers].mean(axis=1)

    return smb, hml


def run_fama_french(returns: pd.DataFrame, mkt_returns: pd.Series, rf_annual: float) -> dict | None:
    if len(returns.columns) < 3:
        return None

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    smb, hml = _build_proxy_factors(returns, mkt_returns, rf_daily)

    if smb is None:
        return None

    mkt_excess = mkt_returns - rf_daily
    results = {}

    for ticker in returns.columns:
        asset_excess = returns[ticker] - rf_daily

        df = pd.DataFrame({
            "y": asset_excess,
            "mkt": mkt_excess,
            "smb": smb,
            "hml": hml,
        }).dropna()

        X = add_constant(df[["mkt", "smb", "hml"]])
        model = OLS(df["y"], X).fit(cov_type="HC3")

        results[ticker] = {
            "Alpha": round(model.params["const"], 6),
            "Beta_Mkt": round(model.params["mkt"], 4),
            "Beta_SMB": round(model.params["smb"], 4),
            "Beta_HML": round(model.params["hml"], 4),
            "R²": round(model.rsquared, 4),
            "R² Ajustado": round(model.rsquared_adj, 4),
            "p_alpha": round(model.pvalues["const"], 4),
            "p_mkt": round(model.pvalues["mkt"], 4),
            "p_smb": round(model.pvalues["smb"], 4),
            "p_hml": round(model.pvalues["hml"], 4),
        }

    return results