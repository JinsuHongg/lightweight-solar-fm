# OpenCode Agent Instructions for lw-solar-fm

This file provides guidance for OpenCode agents working on the `lightweight-solar-fm` repository.

## Project Overview

- **Frameworks:** The project utilizes `pytorch_lightning` for training and `hydra` for configuration management.
- **Execution Entrypoints:** Training and fine-tuning jobs are initiated via Python scripts in the `scripts/` directory (e.g., `scripts/pretraining.py`, `scripts/finetuning.py`).
- **Data Handling:** Data is managed using custom `Dataset` and `DataModule` classes, with data loaded from Zarr stores.
- **Model Architecture:** The models appear to be based on a Vision Transformer (ViT) architecture.

## Development Workflow & Commands

- **Running Training/Finetuning:** Execute scripts directly, e.g., `python scripts/pretraining.py` or `python scripts/finetuning.py`. Configuration is managed via Hydra.
- **Testing:** Explicit commands for running unit or integration tests were not found. Agents should first attempt to identify test files (e.g., `*.test.py`, `tests/`) and then infer execution commands. If no tests are apparent, this should be noted.
- **Linting/Formatting:** No explicit linters (e.g., `ruff`, `flake8`, `mypy`) or formatters were detected. Agents should search for configuration files related to these tools (e.g., `pyproject.toml`, `.eslintrc.*`) or assume standard Python practices if none are found.

## Important Constraints & Quirks

- **No `package.json`:** This project does not appear to use `package.json`, suggesting it's not a standard Node.js project or uses a different build system.
- **Limited `README.md`:** The `README.md` file is minimal and does not provide detailed setup or development instructions.
- **Configuration:** Configuration is handled by Hydra, with configurations likely located in `configs/` directories.

## How to Investigate

1.  **Configuration:** Look for Hydra configuration files (e.g., `*.yaml`) in `configs/` directories.
2.  **Entrypoints:** Examine Python scripts in the `scripts/` directory for execution logic.
3.  **Code Structure:** Understand the modules within `src/` for core components.
4.  **Testing/Linting:** If test or lint commands are not obvious, search for test files (`*_test.py`, `tests/`) and linting configuration files (`pyproject.toml`, `.flake8`, `.pylintrc`). If none are found, note this limitation.
