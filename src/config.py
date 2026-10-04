"""Project-wide settings. Every value here is referenced in the synopsis.

Paths default to ./data inside the repository. On Google Colab, set the
environment variable VS_DATA_ROOT to a Google Drive folder before importing,
for example: os.environ["VS_DATA_ROOT"] = "/content/drive/MyDrive/audio-detection-human-vs-ai"
"""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("VS_DATA_ROOT", REPO_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw" / "asvspoof2019_LA"
CLEAN_DIR = DATA_ROOT / "processed" / "clean"
PHONE_DIR = DATA_ROOT / "processed" / "telephone"
META_DIR = DATA_ROOT / "metadata"
FEAT_DIR = DATA_ROOT / "features"
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"

# ---- Source data (Edinburgh DataShare, doi:10.7488/ds/2555)
LA_ZIP_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
DATASET_DOI = "https://doi.org/10.7488/ds/2555"

# ---- Reproducibility
SEED = 42

# ---- Audio standardisation
SAMPLE_RATE = 16_000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
TRIM_TOP_DB = 30          # silence below (peak - 30 dB) at start/end is trimmed
TARGET_RMS = 0.05         # RMS level after normalisation

# ---- Log-mel spectrogram for the CNN (25 ms window, 10 ms hop, 128 bands)
N_MELS = 128
WIN_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512

# ---- Telephone simulation (RQ4)
PHONE_SR = 8_000
PHONE_BAND = (300.0, 3400.0)
MU = 255

# ---- Planned study sample (synopsis Table 5)
KNOWN_TRAIN_SYSTEMS = ["A01", "A02", "A03", "A04", "A05", "A06"]
KNOWN_EVAL_SYSTEMS = ["A16", "A19"]
UNSEEN_EVAL_SYSTEMS = ["A07", "A08", "A09", "A10", "A11", "A12", "A13",
                       "A14", "A15", "A17", "A18"]
SAMPLE_PLAN = {
    "training":   {"partition": "train", "human": 2580, "per_system": 430, "systems": KNOWN_TRAIN_SYSTEMS},
    "validation": {"partition": "dev",   "human": 1020, "per_system": 170, "systems": KNOWN_TRAIN_SYSTEMS},
    "test_human": {"partition": "eval",  "human": 1500, "per_system": 0,   "systems": []},
    "test_known": {"partition": "eval",  "human": 0,    "per_system": 500, "systems": KNOWN_EVAL_SYSTEMS},
    "test_unknown": {"partition": "eval", "human": 0,   "per_system": 170, "systems": UNSEEN_EVAL_SYSTEMS},
}

# Smoke-test mode: a tiny plan used only to check that the notebooks run end to end.
SMOKE_TEST = os.environ.get("VS_SMOKE_TEST") == "1"
if SMOKE_TEST:
    SAMPLE_PLAN = {k: {**v, "human": min(v["human"], 12), "per_system": min(v["per_system"], 4)}
                   for k, v in SAMPLE_PLAN.items()}

# ---- Business-cost inputs for RQ4 (illustrative, INR per event)
COST_FN = 50_000
COST_FP = 200
COST_REVIEW = 100
PREVALENCES = [0.01, 0.05]


def ensure_dirs():
    for d in [RAW_DIR, CLEAN_DIR, PHONE_DIR, META_DIR, FEAT_DIR, MODEL_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
