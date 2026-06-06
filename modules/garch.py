import pandas as pd
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def run_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict | None:
    """
    Fit a GARCH(p,q) model to the return series.
    Returns dict with fitted model metrics and conditional volatility.
    """
    try:
        # Scale returns to percentage (improves numerical stability)
        r = returns * 100

        model = arch_model(r, vol="Garch", p=p, q=q, dist="normal", rescale=False)
        res = model.fit(disp="off", show_warning=False)

        cond_vol = res.conditional_volatility / 100  # back to decimal

        # Persistence = sum of alpha + beta coefficients
        params = res.params
        alpha_sum = sum(params.get(f"alpha[{i}]", 0) for i in range(1, p + 1))
        beta_sum = sum(params.get(f"beta[{i}]", 0) for i in range(1, q + 1))
        persistence = alpha_sum + beta_sum

        # Parameter table
        summary = pd.DataFrame({
            "Coeficiente": res.params,
            "Erro Padrão": res.std_err,
            "t-stat": res.tvalues,
            "p-valor": res.pvalues,
        })

        # 10-day volatility forecast
        forecast = res.forecast(horizon=10, reindex=False)
        forecast_var = forecast.variance.iloc[-1].values
        forecast_vol = np.sqrt(forecast_var) / 100  # decimal

        return {
            "log_likelihood": round(res.loglikelihood, 4),
            "aic": round(res.aic, 4),
            "bic": round(res.bic, 4),
            "persistence": round(persistence, 6),
            "conditional_vol": cond_vol,
            "params": summary,
            "std_resid": res.std_resid.dropna(),
            "forecast_vol": forecast_vol,
        }

    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return None