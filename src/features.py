"""64 clip-level acoustic features (synopsis Table 6) and log-mel spectrograms for the CNN."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from .preprocess import load

N_MFCC = 20


def feature_names() -> list[str]:
    names = [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    names += [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    for f in ["centroid", "bandwidth", "rolloff", "flatness", "zcr"]:
        names += [f"{f}_mean", f"{f}_std"]
    names += [f"contrast_{i}_mean" for i in range(1, 8)]
    names += ["rms_mean", "rms_std", "rms_dynamic_range_db"]
    names += ["f0_mean", "f0_std", "voiced_fraction", "pause_proportion"]
    return names


FEATURES = feature_names()
assert len(FEATURES) == 64


def extract(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, win_length=C.WIN_LENGTH))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=C.N_FFT,
                                hop_length=C.HOP_LENGTH, win_length=C.WIN_LENGTH)
    spec = [
        librosa.feature.spectral_centroid(S=S, sr=sr)[0],          # Hz
        librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],         # Hz
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],  # Hz
        librosa.feature.spectral_flatness(S=S)[0],                 # 0-1
        librosa.feature.zero_crossing_rate(y, frame_length=C.WIN_LENGTH, hop_length=C.HOP_LENGTH)[0],
    ]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=100.0)  # 7 x T, dB
    rms = librosa.feature.rms(S=S, frame_length=C.N_FFT)[0]
    eps = 1e-10
    p5, p95 = np.percentile(rms, [5, 95])
    dyn_range = 20 * np.log10((p95 + eps) / (p5 + eps))
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=C.HOP_LENGTH)
    f0v = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    rms_db = 20 * np.log10(rms + eps)
    pause = float(np.mean(rms_db < rms_db.max() - 30))          # frames 30 dB below the loudest

    vec = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        np.ravel([[s.mean(), s.std()] for s in spec]),
        contrast.mean(axis=1),
        [rms.mean(), rms.std(), dyn_range],
        [np.nanmean(f0v), np.nanstd(f0v), float(np.mean(voiced)) if voiced is not None else 0.0, pause],
    ])
    return vec.astype(np.float32)


def logmel(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                                       win_length=C.WIN_LENGTH, n_mels=C.N_MELS)
    return librosa.power_to_db(m, ref=1.0).astype(np.float16)   # 128 x 401 for a 4 s clip


def _one(path: Path, with_logmel: bool):
    y = load(path)
    return extract(y), (logmel(y) if with_logmel else None)


def build_feature_table(clip_ids: list[str], audio_dir: Path, with_logmel: bool = True,
                        n_jobs: int = -1) -> tuple[pd.DataFrame, np.ndarray | None]:
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(audio_dir / f"{c}.wav", with_logmel) for c in clip_ids)
    X = pd.DataFrame(np.vstack([r[0] for r in res]), columns=FEATURES)
    X.insert(0, "clip_id", clip_ids)
    L = np.stack([r[1] for r in res]) if with_logmel else None
    return X, L
