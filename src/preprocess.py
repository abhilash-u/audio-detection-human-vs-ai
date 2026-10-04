"""Audio standardisation (silence trim, fixed length, RMS) and telephone-channel simulation."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import config as C


def load(path: str | Path, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def standardise(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Trim leading/trailing silence, crop or zero-pad to CLIP_SECONDS, RMS-normalise.

    Returns the processed clip and QC facts used by the shortcut check.
    Trimming matters: in ASVspoof 2019 LA the silence length alone reveals the label
    (Müller et al., 2021).
    """
    _, (start, end) = librosa.effects.trim(y, top_db=C.TRIM_TOP_DB)
    speech = y[start:end]
    info = {
        "duration_raw_s": len(y) / sr,
        "leading_silence_s": start / sr,
        "trailing_silence_s": (len(y) - end) / sr,
        "duration_trimmed_s": len(speech) / sr,
    }
    speech = speech[: C.CLIP_SAMPLES]
    rms = np.sqrt(np.mean(speech ** 2)) if len(speech) else 0.0
    if rms > 0:
        speech = speech * (C.TARGET_RMS / rms)
    speech = np.clip(speech, -1.0, 1.0).astype(np.float32)
    if len(speech) == 0:
        return np.zeros(C.CLIP_SAMPLES, dtype=np.float32), info
    # Repeat-pad short clips instead of zero-padding: trailing zeros would encode
    # the speech duration and give the models another length shortcut.
    reps = int(np.ceil(C.CLIP_SAMPLES / len(speech)))
    out = np.tile(speech, reps)[: C.CLIP_SAMPLES]
    return out, info


def mu_law(x: np.ndarray, mu: int = C.MU) -> np.ndarray:
    """8-bit mu-law companding (encode, quantise to 256 levels, decode).

    A continuous-curve approximation of the segmented G.711 mu-law codec.
    """
    x = np.clip(x, -1.0, 1.0)
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * mu).astype(np.int32)          # 0..255
    dec_in = 2 * (q.astype(np.float32) / mu) - 1
    return (np.sign(dec_in) * ((1 + mu) ** np.abs(dec_in) - 1) / mu).astype(np.float32)


def telephone(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Simulate a narrowband phone call: 8 kHz, 300-3,400 Hz band-pass, mu-law, back to 16 kHz."""
    y8 = resample_poly(y, C.PHONE_SR, sr).astype(np.float32)
    sos = butter(4, C.PHONE_BAND, btype="bandpass", fs=C.PHONE_SR, output="sos")
    y8 = sosfiltfilt(sos, y8).astype(np.float32)
    y8 = mu_law(y8)
    return resample_poly(y8, sr, C.PHONE_SR).astype(np.float32)[: len(y)]


def _process_one(row, clean_dir: Path, phone_dir: Path | None) -> dict:
    y = load(row["path"])
    clip, info = standardise(y)
    sf.write(clean_dir / f"{row['clip_id']}.wav", clip, C.SAMPLE_RATE, subtype="PCM_16")
    if phone_dir is not None:
        sf.write(phone_dir / f"{row['clip_id']}.wav", telephone(clip), C.SAMPLE_RATE, subtype="PCM_16")
    return {"clip_id": row["clip_id"], **info}


def process_sample(sample: pd.DataFrame, clean_dir: Path = C.CLEAN_DIR,
                   phone_dir: Path = C.PHONE_DIR, n_jobs: int = -1) -> pd.DataFrame:
    """Standardise every sampled clip; make telephone copies of the test clips only."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    phone_dir.mkdir(parents=True, exist_ok=True)
    rows = sample.to_dict("records")
    is_test = sample["split"].str.startswith("test").tolist()
    out = Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(r, clean_dir, phone_dir if t else None) for r, t in zip(rows, is_test))
    qc = pd.DataFrame(out)
    return sample.merge(qc, on="clip_id", how="left")


def clip_path(clip_id: str, channel: str = "clean") -> Path:
    return (C.CLEAN_DIR if channel == "clean" else C.PHONE_DIR) / f"{clip_id}.wav"


def processed_silence(clip_ids: list[str], audio_dir: Path = C.CLEAN_DIR, n_jobs: int = -1) -> pd.DataFrame:
    """Leading/trailing silence re-measured on processed clips (for the shortcut check)."""
    def one(c):
        y = load(audio_dir / f"{c}.wav")
        _, (a, b) = librosa.effects.trim(y, top_db=C.TRIM_TOP_DB)
        return {"clip_id": c, "proc_leading_silence_s": a / C.SAMPLE_RATE,
                "proc_trailing_silence_s": (len(y) - b) / C.SAMPLE_RATE}
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in clip_ids))
