"""Inference prototype: score one audio file and map it to a risk band.

Usage (after notebook 05 has saved models/deployment.joblib):
    python -m src.predict path/to/utterance.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .features import FEATURES, extract, logmel
from .preprocess import load, standardise, telephone


def load_bundle(path: Path = C.MODEL_DIR / "deployment.joblib") -> dict:
    return joblib.load(path)


def score_file(path: str | Path, bundle: dict) -> dict:
    clip, info = standardise(load(path))
    if bundle.get("channel") == "telephone" and bundle.get("simulate_channel", False):
        clip = telephone(clip)
    if bundle["kind"] == "classical":
        x = extract(clip)[None, :]
        import pandas as pd
        score = float(bundle["model"].predict_proba(pd.DataFrame(x, columns=FEATURES))[:, 1][0])
    else:
        from .models import build_cnn, predict_cnn
        import torch
        model = build_cnn(); model.load_state_dict(bundle["state_dict"]); model.eval()
        score = float(predict_cnn(model, logmel(clip)[None], bundle["mean"], bundle["std"])[0])
    t_review, t_high = bundle["thresholds"]["review"], bundle["thresholds"]["high"]
    band = "High" if score >= t_high else ("Review" if score >= t_review else "Low")
    return {"file": str(path), "score_synthetic": round(score, 4), "risk_band": band, **info}


if __name__ == "__main__":
    b = load_bundle()
    for f in sys.argv[1:]:
        print(score_file(f, b))
