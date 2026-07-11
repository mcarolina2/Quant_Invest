import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── Metrics ─────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else 0.0
    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Acurácia Direcional": round(dir_acc, 4),
        "Correlação IC": round(corr, 4),
    }

# ─── Sequence builder ────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ─── PyTorch model ───────────────────────────────────────────────────────────

if HAS_TORCH:
    class _RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, arch="LSTM", dropout=0.2):
            super().__init__()
            rnn_cls = nn.LSTM if arch == "LSTM" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)


def _train_rnn(X_train_sc, y_train, X_test_sc, y_test,
               arch: str = "LSTM",
               seq_len: int = 20,
               hidden_size: int = 64,
               num_layers: int = 2,
               epochs: int = 50,
               batch_size: int = 32,
               lr: float = 1e-3,
               progress_cb=None) -> dict | None:

    if not HAS_TORCH:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs_tr, ys_tr = _make_sequences(X_train_sc, y_train, seq_len)
    Xs_te, ys_te = _make_sequences(X_test_sc, y_test, seq_len)

    if len(Xs_tr) < batch_size:
        batch_size = max(8, len(Xs_tr) // 4)

    ds = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(ys_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = _RNNModel(
        input_size=X_train_sc.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        arch=arch,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        if progress_cb:
            progress_cb(epoch + 1, epochs, epoch_loss)

    model.eval()
    with torch.no_grad():
        preds_test = model(torch.from_numpy(Xs_te).to(device)).cpu().numpy()
        preds_train = model(torch.from_numpy(Xs_tr).to(device)).cpu().numpy()

    # Align targets with sequences
    y_test_aligned = ys_te
    y_train_aligned = ys_tr

    return {
        "arch": arch,
        "preds_test": preds_test,
        "preds_train": preds_train,
        "y_test": y_test_aligned,
        "y_train": y_train_aligned,
        "train_losses": train_losses,
        "test_metrics": _metrics(y_test_aligned, preds_test),
        "train_metrics": _metrics(y_train_aligned, preds_train),
        "model": model,
        "seq_len": seq_len,
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def run_lstm(X_train, y_train, X_test, y_test, seq_len=20, hidden_size=64, num_layers=2, epochs=50, batch_size=32, lr=1e-3, progress_cb=None) -> dict | None:
    if not HAS_TORCH:
        return None
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return _train_rnn(X_tr_sc, y_train.values, X_te_sc, y_test.values,
                      arch="LSTM", seq_len=seq_len, hidden_size=hidden_size,
                      num_layers=num_layers, epochs=epochs, batch_size=batch_size,
                      lr=lr, progress_cb=progress_cb)

def run_gru(X_train, y_train, X_test, y_test, seq_len=20, hidden_size=64, num_layers=2, epochs=50, batch_size=32, lr=1e-3, progress_cb=None) -> dict | None:
    if not HAS_TORCH:
        return None
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return _train_rnn(X_tr_sc, y_train.values, X_te_sc, y_test.values,
                      arch="GRU", seq_len=seq_len, hidden_size=hidden_size,
                      num_layers=num_layers, epochs=epochs, batch_size=batch_size,
                      lr=lr, progress_cb=progress_cb)
