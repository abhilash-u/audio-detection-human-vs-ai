# Voice Spoofing: Human vs AI-Generated Speech Detection

Code for the QM640 Data Analytics Capstone at Walsh College, *Audio-Based Detection of AI-Generated (Synthetic) Voices Versus Genuine Human Voices for Contact-Center Fraud Screening*.

The project classifies short speech clips as **human (bona fide)** or **synthetic (TTS / voice conversion)** using the ASVspoof 2019 Logical Access benchmark, measures how well detection holds up on unseen voice generators and telephone audio, and turns model scores into a cost-optimal risk-screening rule.

## Research questions

| RQ | Question | Notebook |
|---|---|---|
| RQ1 | Which acoustic features differ between human and synthetic speech, and by how much? | `02_features_and_rq1` |
| RQ2 | How accurate are LR, SVM, RF and XGBoost, and does a lightweight CNN do significantly better? | `03_rq2_classical_and_cnn` |
| RQ3 | How much does performance fall on unseen generators, and do one-class detectors generalise better? | `04_rq3_unseen_generators` |
| RQ4 | How much does performance drop on telephone audio, and which threshold minimises expected cost? | `05_rq4_telephone_and_cost` |

Sample sizes for all four RQs are computed in `00_sample_size`. The final sample size is **N = max(RQ1–RQ4) = 4,284 clips** (set by RQ1). The study draws 11,570 clips across disjoint training, validation and test subsets.

## Data

**ASVspoof 2019 Logical Access** — University of Edinburgh, Centre for Speech Technology Research (CSTR).
- Record: https://datashare.ed.ac.uk/handle/10283/3336 · DOI: https://doi.org/10.7488/ds/2555
- File: `LA.zip`, 7.1 GB, no registration required.
- Licence: see the licence file distributed with the data (non-commercial research use).

The raw audio is **not** stored in this repository. Notebook 01 downloads it, records a SHA-256 checksum in `data/metadata/dataset_inventory.csv`, and rebuilds every derived file.

### Planned study sample

| Subset | From partition | Human | Synthetic | Total | Used for |
|---|---|---|---|---|---|
| Training | train | 2,580 | 2,580 (430 × A01–A06) | 5,160 | Model fitting; RQ1 |
| Validation | dev | 1,020 | 1,020 (170 × A01–A06) | 2,040 | Tuning; thresholds; RQ1 |
| Test, human | eval | 1,500 | – | 1,500 | Negative class of both test sets |
| Test-Known | eval | (shared) | 1,000 (500 × A16, A19) | 1,000 | RQ2, RQ4 |
| Test-Unknown | eval | (shared) | 1,870 (170 × 11 unseen systems) | 1,870 | RQ3, RQ4 |

All splits are speaker-disjoint (official ASVspoof partitions) and drawn with a fixed seed (42). `data/metadata/split_assignments.csv` lists every clip used.

### Data dictionary

[`docs/data_dictionary.csv`](docs/data_dictionary.csv) defines every variable used in the study: identifiers, the label (dependent variable), grouping factors, the 64 acoustic features (independent variables) with their units, model outputs and the RQ4 cost parameters.

## How to run

### Option A — Google Colab (recommended)

1. Open a notebook from this repository in Colab (File → Open notebook → GitHub → `abhilash-u/audio-detection-human-vs-ai`).
2. Select a GPU runtime (Runtime → Change runtime type → GPU).
3. Run all cells. The first cell mounts Google Drive, clones the repo and installs the requirements. Data are stored in `MyDrive/audio-detection-human-vs-ai` so they persist between sessions.
4. Run the notebooks in order: `00` → `01` → `02` → `03` → `04` → `05`.

### Option B — local machine

```bash
git clone https://github.com/abhilash-u/audio-detection-human-vs-ai
cd audio-detection-human-vs-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # then run notebooks/00 ... 05 in order
```

Set `VS_DATA_ROOT` to store the data somewhere other than `./data`.

### Approximate runtimes (Colab)

| Step | Time |
|---|---|
| Download and extract LA.zip | 10–20 min |
| Preprocessing (01) | 10–15 min |
| Feature extraction (02) | 30–60 min (pYIN pitch tracking dominates) |
| Classical models + CNN (03) | 20–40 min on GPU |
| RQ3 and RQ4 (04, 05) | < 10 min each |

## Method summary

**Preprocessing** (`src/preprocess.py`): mono 16 kHz → trim leading/trailing silence (30 dB below peak) → crop to 4 s, or repeat-pad shorter clips → RMS normalisation. Silence trimming is essential because silence length alone reveals the label in this dataset (Müller et al., 2021). Notebook 01 runs a **shortcut check**: a classifier that sees only silence lengths must fall to about 0.5 AUC after preprocessing. Short clips are repeat-padded rather than zero-padded so that trailing zeros cannot encode speech length.

**Telephone simulation** (RQ4): resample to 8 kHz → 300–3,400 Hz band-pass → 8-bit µ-law companding (G.711-style) → resample to 16 kHz.

**Features** (`src/features.py`): 64 clip-level features — 20 MFCC means and standard deviations (40); spectral centroid, bandwidth, roll-off, flatness and zero-crossing rate, mean and std (10); spectral contrast in 7 bands (7); RMS mean, std and dynamic range (3); F0 mean and std (pYIN), voiced fraction and internal-pause proportion (4). The CNN uses 128 × 401 log-mel spectrograms (25 ms window, 10 ms hop).

**Models** (`src/models.py`): LR, SVM (RBF), RF and XGBoost tuned on the validation subset by EER; a 4-block CNN trained with Adam and early stopping on validation EER; Isolation Forest and One-Class SVM trained on human clips only.

**Evaluation** (`src/evaluation.py`): accuracy, precision, recall, F1, FPR, ROC-AUC, PR-AUC and EER; 95% speaker-cluster bootstrap CIs; McNemar tests for paired comparisons; Welch t, Mann–Whitney U, Cohen's d and Cliff's delta with Benjamini–Hochberg FDR for RQ1; precision at realistic prevalence; expected cost per 1,000 utterances. Thresholds are always chosen on validation data and applied once to the test sets.

**Business costs (illustrative, RQ4)**: missed fraud ₹50,000, false alarm ₹200, manual review ₹100, at 1% and 5% prevalence; each is varied ±50% in a sensitivity analysis. Edit them in `src/config.py`.

## Inference prototype

After notebook 05 has saved `models/deployment.joblib`:

```bash
python -m src.predict path/to/utterance.wav
# {'score_synthetic': 0.87, 'risk_band': 'High', ...}
```

Risk bands: **Low** (proceed), **Review** (step-up verification before any high-risk action), **High** (route to the fraud team). The tool supports decisions and should never deny service by itself.

## Repository structure

```
audio-detection-human-vs-ai/
├── README.md
├── requirements.txt
├── LICENSE
├── data/                       created by notebook 01 (not committed)
│   ├── raw/asvspoof2019_LA/    downloaded LA.zip and extracted audio
│   ├── processed/clean/        trimmed, 4 s, normalised audio
│   ├── processed/telephone/    RQ4 telephone copies of test clips
│   ├── metadata/               dataset_inventory.csv, audio_metadata.csv, split_assignments.csv
│   └── features/               64-feature tables, log-mel arrays
├── notebooks/
│   ├── 00_sample_size.ipynb
│   ├── 01_download_and_preprocess.ipynb
│   ├── 02_features_and_rq1.ipynb
│   ├── 03_rq2_classical_and_cnn.ipynb
│   ├── 04_rq3_unseen_generators.ipynb
│   └── 05_rq4_telephone_and_cost.ipynb
├── src/                        config, data, preprocess, features, models, evaluation, sample_size, predict
├── tests/                      pytest checks (sample sizes, metrics, end-to-end on toy audio)
├── docs/                       data_dictionary.csv (every variable: type, units, meaning, role)
├── models/                     saved models and deployment bundle
└── reports/                    result tables and figures written by the notebooks
```

### Metadata fields (`split_assignments.csv`)

`clip_id, speaker_id, attack_system_id (A01–A19, blank for human), key, label (0 = human, 1 = synthetic), official_partition, attack_type (TTS / VC / hybrid), system_status (known / unseen), path, source_dataset, source_url, split, duration_raw_s, leading_silence_s, trailing_silence_s, duration_trimmed_s, channel_available`

## Tests

```bash
pytest -q
```

The tests check that the code reproduces the synopsis sample sizes (4,284 / 1,801 / 2,739 / 1,801) and the illustrative metric table. They also run the full pipeline on a small synthetic dataset. To smoke-test every notebook without downloading the real data, point `VS_DATA_ROOT` at a folder containing a toy `raw/asvspoof2019_LA/LA` layout and set `VS_SMOKE_TEST=1`.

## Limitations

The data are read English speech (mostly British accents), not real contact-center calls. The 2019 generators predate current zero-shot voice cloning. The telephone channel is simulated, and results apply to single utterances, not whole calls. If the estimated intra-class correlation is above about 0.09, the RQ1 requirement exceeds the planned pool; notebook 00 shows the sensitivity.

## Key references

- Wang, X., et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language, 64*, 101114. https://doi.org/10.1016/j.csl.2020.101114
- Müller, N. M., et al. (2021). Speech is silver, silence is golden: What do ASVspoof-trained models really learn? https://arxiv.org/abs/2106.12914
- Müller, N. M., et al. (2022). Does audio deepfake detection generalize? *Interspeech 2022*. https://doi.org/10.21437/Interspeech.2022-108
- Connor, R. J. (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics, 43*(1), 207–211.
