import numpy as np
import pandas as pd

from scipy.stats import jarque_bera
from scipy.stats import norm

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)


def compute_risk_metrics(
    returns: pd.Series,
    rf_annual: float
):

    rf_daily = (1 + rf_annual)**(1/252) - 1

    mean_ret = returns.mean()
    std_ret = returns.std()

    # Coeficiente de Variação
    cv = np.nan

    if mean_ret != 0:
        cv = std_ret / abs(mean_ret)

    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(
        returns.dropna()
    )

    # Ljung-Box
    lb = acorr_ljungbox(
        returns.dropna(),
        lags=[10],
        return_df=True
    )

    lb_p = lb["lb_pvalue"].iloc[0]

    # ARCH Test
    arch_stat, arch_p, _, _ = het_arch(
        returns.dropna()
    )

    # VaR
    var95 = np.percentile(
        returns,
        5
    )

    # CVaR
    cvar95 = returns[
        returns <= var95
    ].mean()

    # Sortino
    downside = returns[
        returns < rf_daily
    ]

    if len(downside) > 0:
        sortino = (
            mean_ret - rf_daily
        ) / downside.std()
    else:
        sortino = np.nan

    return {
        "CV": cv,
        "Jarque-Bera": jb_stat,
        "JB p-value": jb_p,
        "Ljung-Box p-value": lb_p,
        "ARCH p-value": arch_p,
        "VaR 95%": var95,
        "CVaR 95%": cvar95,
        "Sortino": sortino
    }