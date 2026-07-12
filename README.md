# Brain Age

Kaggle competition: [eth-fdd-competition](https://www.kaggle.com/competitions/eth-fdd-competition/overview) — brain age prediction. Team project, ML-ETHZ-Project org.

## Team

- Wojciech Bentkowski — wbentkowski@student.ethz.ch
- André Waser — wasera@student.ethz.ch
- Philippe Haas — haasph@student.ethz.ch

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the Jupyter kernel for this venv so notebooks can pick it up:

```bash
python -m ipykernel install --user --name=brain-age --display-name="Python (brain-age)"
```

### Kaggle API

1. Get an API token: Kaggle → your profile → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and lock down permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Never commit `kaggle.json` — it's gitignored, but double-check before pushing.

Download the competition data into `data/raw/`:

```bash
scripts/download_data.sh
```

Submit a predictions file:

```bash
scripts/submit.sh submissions/my_submission.csv "short description of the approach"
```

## Structure

- `data/` — raw/processed data (gitignored, never commit datasets)
- `notebooks/` — exploration notebooks, one per person/experiment
- `src/` — reusable code (features, models, utils) once stable
- `scripts/` — Kaggle CLI helpers (download data, submit predictions)
- `submissions/` — local submission files (gitignored)

## Team workflow

- Don't commit directly to `main`. Work on a branch per person/experiment (e.g. `andre/xgboost-baseline`), open a PR, and get at least one teammate to review before merging.
- Keep `main` in a working state — anything merged should run end to end.
- One notebook per person/experiment under `notebooks/`; once an approach is worth reusing, move the code into `src/`.
- After a training run that's worth recording, add a row to the experiment log below (in the same PR).

## Experiment log

| Date | Who | Approach | CV score | Public LB | Notes |
|------|-----|----------|----------|-----------|-------|
|      |     |          |          |           |       |

## Current best

- Approach:
- CV score:
- Public LB score:
