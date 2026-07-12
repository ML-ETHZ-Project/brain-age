# Brain Age — Kaggle competition

3-person team project (Wojciech Bentkowski, André Waser, Philippe Haas) for the
[eth-fdd-competition](https://www.kaggle.com/competitions/eth-fdd-competition/overview) Kaggle competition.

## On startup

- Check that `.venv/` exists and `requirements.txt` is installed into it; create/install if not.
- Check that a Jupyter kernel is registered for `.venv` (`jupyter kernelspec list`); register it if missing.

## Working conventions

- Never commit real data (`data/`), submission files (`submissions/`), or Kaggle credentials (`kaggle.json`) — all gitignored, double-check before staging.
- Don't commit directly to `main`; use a branch per person/experiment and a PR.
- Notebooks live under `notebooks/`, one per person/experiment. Code worth reusing across notebooks moves into `src/`.
- When a training run produces a result worth keeping, add a row to the experiment log table in `README.md`.
- Use `scripts/download_data.sh` and `scripts/submit.sh` for Kaggle CLI interactions rather than ad hoc `kaggle` commands.
