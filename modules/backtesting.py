import numpy as np
import pandas as pd

def run_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    threshold: float = 0.0,
    transaction_cost: float = 0.001,
    rf_annual: float = 0.1075,
) -> dict:
    n = min(len(y_true), len(y_pred), len(dates))
    y_true = np.array(y_true[:n])
    y_pred = np.array(y_pred[:n])
    dates = dates[:n]

    # Signal: 1 (long) or 0 (flat)
    signal = (y_pred > threshold).astype(float)

    # Position changes for transaction costs
    pos_changes = np.abs(np.diff(signal, prepend=0))
    costs = pos_changes * transaction_cost

    # Strategy returns
    strat_ret = signal * y_true - costs

    # Benchmark: buy-and-hold
    bh_ret = y_true.copy()

    # Cumulative log returns → equity curves
    strat_equity = np.exp(np.cumsum(strat_ret))
    bh_equity = np.exp(np.cumsum(bh_ret))

    # Drawdown
    def _max_drawdown(equity):
        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / roll_max
        return dd

    strat_dd = _max_drawdown(strat_equity)
    bh_dd = _max_drawdown(bh_equity)

    # Performance statistics
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    def _stats(ret_series):
        ann_ret = (1 + ret_series.mean()) ** 252 - 1
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0
        downside = ret_series[ret_series < rf_daily].std() * np.sqrt(252)
        sortino = (ann_ret - rf_annual) / downside if downside > 0 else 0.0
        max_dd = _max_drawdown(np.exp(np.cumsum(ret_series))).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
        win_rate = (ret_series > 0).mean()
        return {
            "Retorno Anualizado": round(ann_ret, 4),
            "Volatilidade Anualizada": round(ann_vol, 4),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "Max Drawdown": round(max_dd, 4),
            "Calmar Ratio": round(calmar, 4),
            "Taxa de Acerto": round(win_rate, 4),
        }

    strat_stats = _stats(strat_ret)
    bh_stats = _stats(bh_ret)
    df_monthly = pd.DataFrame({
        "date": dates,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
    }).set_index("date")
    monthly = df_monthly["strat_ret"].resample("ME").sum()

    return {
        "dates": dates,
        "strat_equity": strat_equity,
        "bh_equity": bh_equity,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "strat_dd": strat_dd,
        "bh_dd": bh_dd,
        "strat_stats": strat_stats,
        "bh_stats": bh_stats,
        "signal": signal,
        "monthly_returns": monthly,
        "n_trades": int(pos_changes.sum()),
    }
