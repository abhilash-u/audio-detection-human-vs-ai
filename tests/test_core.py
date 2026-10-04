"""Fast checks that the code reproduces the synopsis numbers and runs end to end."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src import config as C
from src import data, evaluation as E, features as F, preprocess as P, sample_size as S


def test_sample_sizes_match_synopsis():
    r1, r2, r3, r4 = S.rq1(), S.rq2(), S.rq3(), S.rq4()
    assert (r1["n0_per_group"], r1["n_per_group"], r1["minimum_N"]) == (393, 2142, 4284)
    assert round(r1["detectable_d"], 2) == 0.23
    assert (r2["n0_pairs"], r2["minimum_N"]) == (870, 1801)
    assert (r3["n0_per_system"], r3["n_per_system"], r3["n0_fpr"], r3["n_human"], r3["minimum_N"]) == (151, 163, 457, 946, 2739)
    assert r4["minimum_N"] == 1801
    table, final_n = S.summary()
    assert final_n == 4284 and table.meets_requirement.all()


def test_illustrative_metrics_match_table8():
    y = np.r_[np.ones(1000), np.zeros(1500)].astype(int)
    pred = np.r_[np.ones(930), np.zeros(70), np.ones(90), np.zeros(1410)]
    r = E.classification_report(y, pred, 0.5)
    assert (r["TP"], r["FN"], r["FP"], r["TN"]) == (930, 70, 90, 1410)
    assert round(r["accuracy"], 3) == 0.936 and round(r["f1"], 3) == 0.921
    assert round(E.precision_at_prevalence(0.93, 0.06, 0.01), 3) == 0.135
    assert round(E.expected_cost(0.93, 0.06, 0.01)) == 53750
    m = E.mcnemar(np.r_[np.ones(60), np.zeros(35), np.ones(5)], np.r_[np.zeros(60), np.ones(35), np.ones(5)])
    assert round(m["statistic"], 2) == 6.06 and round(m["p_value"], 3) == 0.014


def test_eer_perfect_and_random():
    y = np.r_[np.zeros(100), np.ones(100)]
    assert E.eer(y, y)[0] == 0
    rng = np.random.default_rng(0)
    assert 0.35 < E.eer(y, rng.random(200))[0] < 0.65


def test_standardise_and_telephone():
    sr = C.SAMPLE_RATE
    t = np.arange(int(1.5 * sr)) / sr
    y = np.r_[np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(sr // 4)].astype(np.float32)
    clip, info = P.standardise(y)
    assert len(clip) == C.CLIP_SAMPLES
    assert abs(info["leading_silence_s"] - 0.5) < 0.1
    assert abs(np.sqrt(np.mean(clip ** 2)) - C.TARGET_RMS) < 0.01
    ph = P.telephone(clip)
    assert len(ph) == len(clip) and np.isfinite(ph).all()
    spec = np.abs(np.fft.rfft(P.telephone(np.random.default_rng(0).normal(0, 0.05, sr).astype(np.float32))))
    freqs = np.fft.rfftfreq(sr, 1 / sr)
    assert spec[freqs > 5000].mean() < 0.05 * spec[(freqs > 500) & (freqs < 3000)].mean()


def test_feature_vector():
    sr = C.SAMPLE_RATE
    y = (0.1 * np.sin(2 * np.pi * 150 * np.arange(C.CLIP_SAMPLES) / sr)).astype(np.float32)
    v = F.extract(y)
    assert v.shape == (64,) and np.isfinite(v).all()
    assert 130 < v[F.FEATURES.index("f0_mean")] < 170
    assert F.logmel(y).shape == (128, 401)


def _fake_la(root, n_per=6):
    """Tiny fake ASVspoof layout: bona fide = harmonic tones, spoof = noisy tones."""
    rng = np.random.default_rng(1)
    la = root / "LA"
    (la / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    systems = {"train": C.KNOWN_TRAIN_SYSTEMS, "dev": C.KNOWN_TRAIN_SYSTEMS,
               "eval": C.KNOWN_EVAL_SYSTEMS + C.UNSEEN_EVAL_SYSTEMS}
    for part, (proto, adir) in data.PARTITIONS.items():
        (la / adir / "flac").mkdir(parents=True)
        lines = []
        for k in range(n_per * 3):
            spk = f"LA_{part}_{k % 4}"
            cid = f"{part}_bona_{k}"
            f0 = rng.uniform(100, 220)
            t = np.arange(int(rng.uniform(1.5, 3) * 16000)) / 16000
            y = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 5)) * 0.1
            sf.write(la / adir / "flac" / f"{cid}.flac", np.r_[np.zeros(4000), y].astype(np.float32), 16000)
            lines.append(f"{spk} {cid} - - bonafide")
        for s in systems[part]:
            for k in range(n_per):
                spk = f"LA_{part}_{k % 4}"
                cid = f"{part}_{s}_{k}"
                t = np.arange(int(rng.uniform(1.5, 3) * 16000)) / 16000
                y = 0.1 * np.sin(2 * np.pi * rng.uniform(100, 220) * t) + rng.normal(0, 0.03, len(t))
                sf.write(la / adir / "flac" / f"{cid}.flac", y.astype(np.float32), 16000)
                lines.append(f"{spk} {cid} - {s} spoof")
        (la / "ASVspoof2019_LA_cm_protocols" / proto).write_text("\n".join(lines))


def test_pipeline_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    _fake_la(raw)
    meta = data.build_metadata(raw)
    assert set(meta.label) == {0, 1}
    plan = {k: {**v, "human": min(v["human"], 12), "per_system": min(v["per_system"], 4)} for k, v in C.SAMPLE_PLAN.items()}
    sample = data.draw_sample(meta, plan)
    assert set(sample.split) == set(plan)
    clean, phone = tmp_path / "clean", tmp_path / "phone"
    sample = P.process_sample(sample, clean, phone, n_jobs=1)
    assert (sample.leading_silence_s >= 0).all()
    X, L = F.build_feature_table(sample.clip_id.tolist(), clean, n_jobs=1)
    assert X.shape == (len(sample), 65) and L.shape == (len(sample), 128, 401)
    tests = E.feature_tests(pd.concat([sample[["label"]], X[F.FEATURES]], axis=1), F.FEATURES)
    assert len(tests) == 64
    from src import models as M
    tr, va = sample.split == "training", sample.split == "validation"
    pipe, params, v_eer = M.tune_classical("LR", X.loc[tr, F.FEATURES], sample.label[tr], X.loc[va, F.FEATURES], sample.label[va])
    assert v_eer <= 0.5
    model, mu, sd, hist = M.train_cnn(L[tr.values], sample.label[tr].values, L[va.values], sample.label[va].values,
                                      epochs=2, batch=8, verbose=False)
    s = M.predict_cnn(model, L[va.values], mu, sd)
    assert s.shape == (va.sum(),) and ((s >= 0) & (s <= 1)).all()
    oc = M.fit_one_class("IsolationForest", X.loc[tr & (sample.label == 0), F.FEATURES])
    assert np.isfinite(M.one_class_score(oc, X.loc[va, F.FEATURES])).all()
