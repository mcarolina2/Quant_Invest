import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def run_capm(returns: pd.DataFrame, mkt_returns: pd.Series, rf_annual: float) -> dict:
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    results = {}
    mkt_excess = mkt_returns - rf_daily
    mkt_annual_return = (1 + mkt_returns.mean()) ** 252 - 1

    for ticker in returns.columns:
        asset_excess = returns[ticker] - rf_daily
        common_idx = asset_excess.dropna().index.intersection(mkt_excess.dropna().index)

        y = asset_excess.loc[common_idx].values
        x = mkt_excess.loc[common_idx].values
        X = add_constant(x)
        model = OLS(y, X).fit(cov_type="HC3")

        alpha = model.params[0]   # Jensen's alpha (daily)
        beta = model.params[1]
        r2 = model.rsquared
        p_alpha = model.pvalues[0]
        p_beta = model.pvalues[1]
        conf_int = model.conf_int()

        # Annualised metrics
        alpha_annual = (1 + alpha) ** 252 - 1
        risk_premium = beta * (mkt_annual_return - rf_annual)
        expected_return = rf_annual + risk_premium
        realized_return = (1 + returns[ticker].mean()) ** 252 - 1

        try:
            ci_low = float(conf_int[1, 0]) if hasattr(conf_int, '__getitem__') and not hasattr(conf_int, 'iloc') else float(conf_int.iloc[1, 0])
            ci_high = float(conf_int[1, 1]) if hasattr(conf_int, '__getitem__') and not hasattr(conf_int, 'iloc') else float(conf_int.iloc[1, 1])
        except Exception:
            ci_low, ci_high = float("nan"), float("nan")

        results[ticker] = {
            "Alpha (Jensen)": round(alpha_annual, 6),
            "Beta": round(beta, 4),
            "R²": round(r2, 4),
            "p-valor Alpha": round(p_alpha, 4),
            "p-valor Beta": round(p_beta, 4),
            "Beta IC 95% Inf.": round(ci_low, 4),
            "Beta IC 95% Sup.": round(ci_high, 4),
            "Prêmio de Risco": round(risk_premium, 4),
            "Retorno Esperado (CAPM)": round(expected_return, 4),
            "Retorno Realizado": round(realized_return, 4),
        }

    return results