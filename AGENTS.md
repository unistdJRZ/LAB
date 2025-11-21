# Repository Guidelines

## Project Structure & Modules
- `cf_framework/`: core library (`CFModel`, data utilities, config dataclasses).
- `configs/`: YAML configs (model, training, data). Use lower_snake_case filenames.
- `train.py`: CLI entry to train from a YAML config.
- `utils.py`: training helpers (seeding, dataloaders).
- `runs/<exp_name>/`: training outputs (CSV logs, checkpoints, plots).
- `pivots/`: saved pivot assignments (`pivot_<name>_mu<val>_seed<seed>.yaml`).
- `test_datasets.py`: quick sanity script for dataset loading.

## Setup, Build & Run
- Create environment and install deps (uv):
  - `uv sync` (reads `pyproject.toml` + `uv.lock`)
- Run training:
  - `uv run python train.py --config configs/example_torchvision.yaml`
  - Defaults to `configs/default.yaml` if `--config` is omitted.
- Alternative (pip):
  - `python -m venv .venv && .venv\Scripts\activate`
  - `pip install -e .`

## Coding Style & Naming
- Python 3.12+, 4‑space indentation, type hints required for new/changed code.
- Names: modules/functions `lower_snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Config keys in YAML use `lower_snake_case`; place files in `configs/` (e.g., `resnet18_food101d.yaml`).
- Outputs under `runs/<model>_<dataset>/` (e.g., `runs/resnet18_food101d/`).

## Testing Guidelines
- Current tests are lightweight. Use the sanity script:
  - `uv run python test_datasets.py`
- For new features, add focused tests alongside modules or in `tests/` as `test_*.py` (pytest-friendly). Keep runs short (epochs=1, small batch) for speed.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (≤72 chars), explain motivation and impact.
  - Example: `feat(cf_framework): add pivot-balanced sampler`
- PRs must include:
  - Clear description of what/why, linked issues.
  - Config diff or snippet, plus an example run command.
  - Evidence: path to logs/plots (e.g., `runs/<exp>/training_curve.png`).
  - Note any dependency or config changes (`pyproject.toml`, `configs/*`).

## Security & Config Tips
- Prefer CPU-safe defaults; enable GPU via YAML (`training.device: cuda`) when available.
- HuggingFace datasets: cache with `HF_DATASETS_CACHE` and set `data.*` in YAML.
- Optional logging: enable Weights & Biases via `training.wandb: true` and `training.wandb_project`.
