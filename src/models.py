"""Classical models (RQ2), one-class detectors (RQ3) and the lightweight CNN (RQ2)."""
from __future__ import annotations

import time
from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from xgboost import XGBClassifier

from . import config as C
from .evaluation import eer

# ---------------------------------------------------------------- classical (RQ2)
GRIDS = {
    "LR": (lambda p: LogisticRegression(C=p["C"], max_iter=5000),
           {"C": [0.01, 0.1, 1, 10]}),
    "SVM": (lambda p: SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", probability=True, random_state=C.SEED),
            {"C": [1, 10], "gamma": ["scale", 0.01]}),
    "RF": (lambda p: RandomForestClassifier(n_estimators=p["n"], max_depth=p["depth"], n_jobs=-1,
                                            random_state=C.SEED),
           {"n": [300], "depth": [None, 12]}),
    "XGBoost": (lambda p: XGBClassifier(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"],
                                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                        random_state=C.SEED, n_jobs=-1),
                {"n": [300], "depth": [4, 6], "lr": [0.05, 0.1]}),
}


def _pipe(est):
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def tune_classical(name: str, X_tr, y_tr, X_val, y_val) -> tuple[Pipeline, dict, float]:
    """Grid search on the fixed validation subset; select by lowest validation EER."""
    make, grid = GRIDS[name]
    best = (None, None, np.inf)
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        pipe = _pipe(make(params)).fit(X_tr, y_tr)
        e = eer(y_val, pipe.predict_proba(X_val)[:, 1])[0]
        if e < best[2]:
            best = (pipe, params, e)
    return best


# ---------------------------------------------------------------- one-class (RQ3)
def fit_one_class(kind: str, X_human):
    est = (IsolationForest(n_estimators=300, random_state=C.SEED) if kind == "IsolationForest"
           else OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"))
    return _pipe(est).fit(X_human)


def one_class_score(pipe, X) -> np.ndarray:
    """Higher = more anomalous = more likely synthetic."""
    return -pipe.decision_function(X)


# ---------------------------------------------------------------- CNN (RQ2)
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_cnn():
    torch, nn = _torch()

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Dropout(0.1))

    class SpoofCNN(nn.Module):
        """Four conv blocks + global average pooling. Input: (batch, 1, 128, 401) log-mel."""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, x):
            return self.head(self.features(x)).squeeze(1)

    return SpoofCNN()


def _normalise(L: np.ndarray, mean: float, std: float):
    return (L.astype(np.float32) - mean) / std


def predict_cnn(model, L: np.ndarray, mean: float, std: float, batch: int = 64) -> np.ndarray:
    torch, _ = _torch()
    dev = next(model.parameters()).device
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(L), batch):
            x = torch.from_numpy(_normalise(L[i:i + batch], mean, std)).unsqueeze(1).to(dev)
            out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out)


def train_cnn(L_tr, y_tr, L_val, y_val, epochs: int = 30, batch: int = 64, lr: float = 1e-3,
              patience: int = 5, device: str | None = None, verbose: bool = True):
    """Train with Adam + BCE; early-stop on validation EER. Returns (model, mean, std, history)."""
    torch, nn = _torch()
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = float(L_tr.astype(np.float32).mean()), float(L_tr.astype(np.float32).std() + 1e-6)
    model = build_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best_state, best_eer, wait, hist = None, np.inf, 0, []
    rng = np.random.default_rng(C.SEED)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        idx = rng.permutation(len(L_tr)); total = 0.0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            x = torch.from_numpy(_normalise(L_tr[b], mean, std)).unsqueeze(1).to(device)
            y = torch.from_numpy(y_tr[b].astype(np.float32)).to(device)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
            total += loss.item() * len(b)
        v_eer = eer(y_val, predict_cnn(model, L_val, mean, std))[0]
        hist.append({"epoch": ep + 1, "train_loss": total / len(idx), "val_eer": v_eer, "seconds": time.time() - t0})
        if verbose:
            print(f"epoch {ep + 1:2d}  loss {total / len(idx):.4f}  val EER {v_eer:.4f}")
        if v_eer < best_eer:
            best_eer, wait = v_eer, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, hist
