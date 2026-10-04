"""Sample-size calculations for RQ1-RQ4, reproducing synopsis Table 3.

Common assumptions: two-sided alpha = .05, power = .80, 95% confidence,
intra-class correlation rho = .05 (to be re-estimated in Week 3).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

ALPHA = 0.05
POWER = 0.80
RHO = 0.05
Z95 = norm.ppf(0.975)
Z_BETA = norm.ppf(POWER)


def deff(m: float, rho: float = RHO) -> float:
    """Kish design effect for clustered clips: 1 + (m - 1) * rho."""
    return 1 + (m - 1) * rho


def two_means_n(d: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Per-group n for a two-sided two-sample comparison of means (Cohen's d)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return 2 * (za + zb) ** 2 / d ** 2


def mcnemar_pairs(psi: float, delta: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Connor (1987) number of pairs for McNemar's test."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2


def proportion_ci_n(p: float, e: float, z: float = Z95) -> float:
    """Cochran (1977) n for estimating a proportion within +/- e."""
    return z ** 2 * p * (1 - p) / e ** 2


def ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def rq1(rho: float = RHO, n_features: int = 64, d: float = 0.3,
        clips_per_class: int = 3600, speakers: int = 40) -> dict:
    alpha_b = ALPHA / n_features
    n0 = ceil(two_means_n(d, alpha_b))
    m = clips_per_class / speakers
    de = deff(m, rho)
    n_group = ceil(n0 * de)
    n_eff = clips_per_class / de
    mde = (norm.ppf(1 - alpha_b / 2) + Z_BETA) * math.sqrt(2 / n_eff)
    return {"rq": "RQ1", "method": "Power: two independent means", "alpha": alpha_b, "d": d,
            "n0_per_group": n0, "m": m, "deff": de, "n_per_group": n_group,
            "minimum_N": 2 * n_group, "planned": 2 * clips_per_class, "detectable_d": mde}


def rq2(rho: float = RHO, psi: float = 0.10, delta: float = 0.03,
        test_human: int = 1500, eval_speakers: int = 67, planned_pairs: int = 2500) -> dict:
    n0 = ceil(mcnemar_pairs(psi, delta))
    de = deff(test_human / eval_speakers, rho)
    return {"rq": "RQ2", "method": "Power: paired proportions (McNemar)", "psi": psi, "delta": delta,
            "n0_pairs": n0, "deff": de, "minimum_N": ceil(n0 * de), "planned": planned_pairs,
            "epv_training_min_synthetic": 20 * 64}


def rq3(rho: float = RHO, e_sys: float = 0.08, per_system: int = 170, n_systems: int = 11,
        eval_speakers: int = 67, fpr: float = 0.05, e_fpr: float = 0.02, test_human: int = 1500) -> dict:
    n_sys0 = ceil(proportion_ci_n(0.5, e_sys))
    de_sys = deff(per_system / eval_speakers, rho)
    n_sys = ceil(n_sys0 * de_sys)
    n_fpr0 = ceil(proportion_ci_n(fpr, e_fpr))
    de_h = deff(test_human / eval_speakers, rho)
    n_fpr = ceil(n_fpr0 * de_h)
    return {"rq": "RQ3", "method": "Confidence interval for proportions", "n0_per_system": n_sys0,
            "deff_system": de_sys, "n_per_system": n_sys, "n0_fpr": n_fpr0, "deff_human": de_h,
            "n_human": n_fpr, "minimum_N": n_sys * n_systems + n_fpr,
            "planned": per_system * n_systems + test_human}


def rq4(rho: float = RHO, psi: float = 0.10, delta: float = 0.03,
        test_human: int = 1500, eval_speakers: int = 67, planned_pairs: int = 4370) -> dict:
    r = rq2(rho, psi, delta, test_human, eval_speakers, planned_pairs)
    r.update({"rq": "RQ4", "method": "Power: paired proportions (McNemar), clean vs telephone"})
    r.pop("epv_training_min_synthetic")
    return r


def summary(rho: float = RHO) -> tuple[pd.DataFrame, int]:
    rows = [rq1(rho), rq2(rho), rq3(rho), rq4(rho)]
    df = pd.DataFrame([{k: r[k] for k in ["rq", "method", "minimum_N", "planned"]} for r in rows])
    df["meets_requirement"] = df.planned >= df.minimum_N
    return df, int(df.minimum_N.max())


def estimate_icc(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA ICC(1) for a feature clustered by speaker (used to update rho in Week 3)."""
    df = pd.DataFrame({"v": values, "g": groups})
    k = df.groupby("g").size()
    n_groups, N = len(k), len(df)
    k0 = (N - (k ** 2).sum() / N) / (n_groups - 1)
    grand = df.v.mean()
    msb = (k * (df.groupby("g").v.mean() - grand) ** 2).sum() / (n_groups - 1)
    msw = ((df.v - df.groupby("g").v.transform("mean")) ** 2).sum() / (N - n_groups)
    return float((msb - msw) / (msb + (k0 - 1) * msw))
