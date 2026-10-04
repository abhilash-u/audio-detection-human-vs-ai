"""Metrics, statistical tests and the business-cost threshold (synopsis Evaluation Metrics).

Positive class = synthetic (label 1). Scores are P(synthetic).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from . import config as C


# ---------------------------------------------------------------- core metrics
def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where FPR = FNR."""
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def classification_report(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    p = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y), "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall_tpr": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": roc_auc_score(y, s) if len(set(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, s) if len(set(y)) > 1 else np.nan,
        "eer": eer(y, s)[0] if len(set(y)) > 1 else np.nan,
    }


def precision_at_prevalence(tpr: float, fpr: float, pi: float) -> float:
    den = tpr * pi + fpr * (1 - pi)
    return tpr * pi / den if den else 0.0


def expected_cost(tpr: float, fpr: float, pi: float, c_fn: float = C.COST_FN,
                  c_fp: float = C.COST_FP, c_review: float = C.COST_REVIEW, per: int = 1000) -> float:
    """Expected cost per `per` screened utterances (synopsis RQ4 formula)."""
    return per * (pi * (1 - tpr) * c_fn + (1 - pi) * fpr * c_fp + (pi * tpr + (1 - pi) * fpr) * c_review)


def cost_curve(y: np.ndarray, s: np.ndarray, pi: float, **costs) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, s)
    df = pd.DataFrame({"threshold": thr, "tpr": tpr, "fpr": fpr})
    df["expected_cost"] = [expected_cost(t, f, pi, **costs) for t, f in zip(tpr, fpr)]
    df["precision_at_pi"] = [precision_at_prevalence(t, f, pi) for t, f in zip(tpr, fpr)]
    return df[np.isfinite(df.threshold)]


def cost_optimal_threshold(y: np.ndarray, s: np.ndarray, pi: float, **costs) -> dict:
    """Pick the threshold minimising expected cost. Use VALIDATION scores only."""
    df = cost_curve(y, s, pi, **costs)
    best = df.loc[df.expected_cost.idxmin()]
    c_fn = costs.get("c_fn", C.COST_FN)
    return {"prevalence": pi, **best.to_dict(), "cost_no_screening": 1000 * pi * c_fn}


# ---------------------------------------------------------------- paired comparison
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar test on paired correctness. b = A right & B wrong, c = A wrong & B right."""
    a, bb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int(np.sum(a & ~bb)); c = int(np.sum(~a & bb))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    if b + c < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": chi2, "p_value": float(stats.chi2.sf(chi2, 1)),
            "method": "chi-square with continuity correction"}


# ---------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap(df: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                      cluster: str = "speaker_id", n_boot: int = 2000, seed: int = C.SEED,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole speakers with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    est = metric(df)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        try:
            boots.append(metric(sample))
        except ValueError:
            continue
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


# ---------------------------------------------------------------- RQ1 effect sizes
def cohens_d(x_syn: np.ndarray, x_hum: np.ndarray) -> float:
    n1, n2 = len(x_syn), len(x_hum)
    sp = np.sqrt(((n1 - 1) * np.var(x_syn, ddof=1) + (n2 - 1) * np.var(x_hum, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(x_syn) - np.mean(x_hum)) / sp) if sp > 0 else 0.0


def cliffs_delta(x_syn: np.ndarray, x_hum: np.ndarray) -> float:
    u = stats.mannwhitneyu(x_syn, x_hum, alternative="two-sided").statistic
    return float(2 * u / (len(x_syn) * len(x_hum)) - 1)


def feature_tests(df: pd.DataFrame, features: list[str], label: str = "label") -> pd.DataFrame:
    """Welch t-test, Mann-Whitney U, Cohen's d and Cliff's delta per feature, with BH-FDR."""
    rows = []
    syn, hum = df[df[label] == 1], df[df[label] == 0]
    for f in features:
        a, b = syn[f].to_numpy(), hum[f].to_numpy()
        rows.append({
            "feature": f,
            "mean_synthetic": a.mean(), "mean_human": b.mean(),
            "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
            "mwu_p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
            "cohens_d": cohens_d(a, b),
            "cliffs_delta": cliffs_delta(a, b),
        })
    out = pd.DataFrame(rows)
    out["welch_q_bh"] = multipletests(out.welch_p, method="fdr_bh")[1]
    out["mwu_q_bh"] = multipletests(out.mwu_p, method="fdr_bh")[1]
    out["abs_d"] = out.cohens_d.abs()
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- shortcut check
def shortcut_auc(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], label: str = "label") -> float:
    """Validation ROC-AUC of a logistic regression that sees only `cols` (e.g. silence lengths).

    Before trimming this is expected to be high; after trimming it should be close to 0.5.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[cols], train[label])
    return float(roc_auc_score(valid[label], m.predict_proba(valid[cols])[:, 1]))
