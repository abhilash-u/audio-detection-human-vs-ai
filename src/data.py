"""Download ASVspoof 2019 LA, parse its protocol files and draw the study sample."""
from __future__ import annotations

import hashlib
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from . import config as C

PARTITIONS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}

# Generation-system families as described by Wang et al. (2020)
ATTACK_TYPE = {
    "A01": "TTS", "A02": "TTS", "A03": "TTS", "A04": "TTS", "A05": "VC", "A06": "VC",
    "A07": "TTS", "A08": "TTS", "A09": "TTS", "A10": "TTS", "A11": "TTS", "A12": "TTS",
    "A13": "hybrid", "A14": "hybrid", "A15": "hybrid", "A16": "TTS", "A17": "VC",
    "A18": "VC", "A19": "VC",
}


def la_root(raw_dir: Path = C.RAW_DIR) -> Path:
    return raw_dir / "LA"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_la(raw_dir: Path = C.RAW_DIR, url: str = C.LA_ZIP_URL) -> Path:
    """Download LA.zip (7.1 GB) once and extract it into raw_dir/LA. Returns the zip path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "LA.zip"
    if la_root(raw_dir).exists():
        print(f"Found extracted data at {la_root(raw_dir)}; skipping download.")
        return zpath
    if not zpath.exists():
        print(f"Downloading {url} -> {zpath}")
        tmp = zpath.with_suffix(".part")
        with urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while block := r.read(1 << 22):
                f.write(block)
                done += len(block)
                if total and done % (1 << 28) < (1 << 22):
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
        tmp.rename(zpath)
    if not la_root(raw_dir).exists():
        print("Extracting LA.zip ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw_dir)
    return zpath


def read_protocol(partition: str, raw_dir: Path = C.RAW_DIR) -> pd.DataFrame:
    """Parse one official protocol file: SPEAKER CLIP - SYSTEM KEY."""
    proto_name, audio_dir = PARTITIONS[partition]
    path = la_root(raw_dir) / "ASVspoof2019_LA_cm_protocols" / proto_name
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["speaker_id", "clip_id", "unused", "attack_system_id", "key"])
    df = df.drop(columns="unused")
    df["attack_system_id"] = df["attack_system_id"].replace("-", "")
    df["label"] = (df["key"] == "spoof").astype(int)          # 1 = synthetic, 0 = human
    df["official_partition"] = partition
    df["attack_type"] = df["attack_system_id"].map(ATTACK_TYPE).fillna("")
    known = set(C.KNOWN_TRAIN_SYSTEMS + C.KNOWN_EVAL_SYSTEMS)
    df["system_status"] = df["attack_system_id"].apply(
        lambda s: "" if s == "" else ("known" if s in known else "unseen"))
    df["path"] = df["clip_id"].apply(
        lambda c: str(la_root(raw_dir) / audio_dir / "flac" / f"{c}.flac"))
    return df


def build_metadata(raw_dir: Path = C.RAW_DIR) -> pd.DataFrame:
    meta = pd.concat([read_protocol(p, raw_dir) for p in PARTITIONS], ignore_index=True)
    meta["source_dataset"] = "ASVspoof2019_LA"
    meta["source_url"] = C.DATASET_DOI
    return meta


def draw_sample(meta: pd.DataFrame, plan: dict = C.SAMPLE_PLAN, seed: int = C.SEED) -> pd.DataFrame:
    """Draw the stratified, seeded study sample of synopsis Table 5.

    If a stratum has fewer clips than planned (only possible on toy data),
    all available clips are taken and a warning is printed.
    """
    parts = []
    for split, spec in plan.items():
        pool = meta[meta["official_partition"] == spec["partition"]]
        if spec["human"]:
            h = pool[pool["label"] == 0]
            n = min(spec["human"], len(h))
            if n < spec["human"]:
                print(f"WARNING {split}: {n} human clips available, {spec['human']} planned")
            parts.append(h.sample(n=n, random_state=seed).assign(split=split))
        for sys_id in spec["systems"]:
            s = pool[pool["attack_system_id"] == sys_id]
            n = min(spec["per_system"], len(s))
            if n < spec["per_system"]:
                print(f"WARNING {split}/{sys_id}: {n} clips available, {spec['per_system']} planned")
            parts.append(s.sample(n=n, random_state=seed).assign(split=split))
    return pd.concat(parts, ignore_index=True)


def write_inventory(zpath: Path | None, meta: pd.DataFrame, out: Path) -> pd.DataFrame:
    inv = pd.DataFrame([{
        "dataset": "ASVspoof 2019 Logical Access",
        "source_url": C.LA_ZIP_URL,
        "doi": C.DATASET_DOI,
        "download_date": date.today().isoformat(),
        "licence": "See LICENSE file inside LA.zip (non-commercial research use)",
        "n_clips": len(meta),
        "n_human": int((meta.label == 0).sum()),
        "n_synthetic": int((meta.label == 1).sum()),
        "sha256_LA_zip": sha256(zpath) if zpath is not None and zpath.exists() else "",
        "label_definition": "bonafide = human (0); spoof = synthetic (1)",
    }])
    inv.to_csv(out, index=False)
    return inv


def eval_groups(sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the evaluation sets: Test-Known and Test-Unknown each include the shared human clips."""
    human = sample[sample.split == "test_human"]
    return {
        "test_known": pd.concat([human, sample[sample.split == "test_known"]], ignore_index=True),
        "test_unknown": pd.concat([human, sample[sample.split == "test_unknown"]], ignore_index=True),
    }
