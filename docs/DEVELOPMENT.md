# Development / 開発ガイド

## Clone

```bash
mkdir -p ~/projects && cd ~/projects
git clone git@github.com:Wooden243/ml-algorithms.git
cd ml-algorithms
```

GitHub SSH認証が未設定の場合はHTTPSのclone URLを使用してください。

## Setup and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python multi-armed-bandit/experiment.py --n-runs 100 --output /tmp/bandit-preview
# Reproduce full 5,000-run-per-condition experiment:
python multi-armed-bandit/experiment.py --n-runs 5000 --output /tmp/bandit-reproduction
```

### Documentation checks

The pytest suite also checks for known unsupported GitHub math notation and broken relative Markdown links. Read the [bandit reading guide](../multi-armed-bandit/docs/reading-guide.md) before modifying the simulation. Use an explicit `--output` directory for experimental runs to avoid overwriting tracked CSVs and plots.

## Add a new algorithm

1. Make a subdirectory for a *completed* implementation (e.g. `softmax/`).
2. Keep algorithm code, experiments, tests and project README together.
3. Explain mathematical assumptions, dependencies, seeds, evaluation and limitations.
4. Add the new project to root README.
5. Open a PR and run tests before merging.

```bash
git switch -c feat/my-next-implementation
# add code, docs and tests
git add .
git commit -m "Implement and evaluate next algorithm"
git push -u origin feat/my-next-implementation
```

**Public repository note:** Do not commit private keys, personal information, or materials not permitted for public distribution. The original course notebook in this repository was sanitized by removing execution outputs and runtime metadata.
