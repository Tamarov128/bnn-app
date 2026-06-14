# bnn-app

A PyQt6 desktop application and headless CLI for training and evaluating **Bayesian Neural Networks** (BNNs) via MC Dropout on greyscale image classification benchmarks (MNIST, Fashion-MNIST, KMNIST, NotMNIST, Omniglot).

The project combines a full training pipeline with uncertainty quantification — comparing deterministic softmax predictions against Monte Carlo Dropout estimates — and exposes everything through both a polished GUI and a scriptable command-line interface.

---

## Table of Contents

1. [Project Scope](#project-scope)
2. [Project Structure](#project-structure)
3. [Configuration & Setup](#configuration--setup)
   - [Python virtual environment](#1-python-virtual-environment)
   - [Install CUDA (GPU support)](#2-install-cuda-gpu-support)
   - [Install Python requirements](#3-install-python-requirements)
4. [Running a Single Experiment (CLI)](#running-a-single-experiment-cli)
5. [Running the Test Suite](#running-the-test-suite)
6. [Launching the GUI](#launching-the-gui)

---

## Project Scope

`bnn-app` lets you:

- **Train** an AlexNet-inspired CNN on any of the five supported greyscale datasets, with configurable dropout, learning rate, epochs, and training-set fraction.
- **Evaluate uncertainty** using two inference strategies side-by-side:
  - *Deterministic* — standard softmax with `model.eval()` (dropout disabled).
  - *MC Dropout* — stochastic forward passes at inference time to approximate a Bayesian posterior and produce predictive entropy / AUROC-based OOD scores.
- **Detect out-of-distribution samples** by evaluating the trained model against all datasets that were not used for training.
- **Visualise results** interactively through the GUI (live loss/accuracy curves, calibration plots, OOD metrics, a freehand drawing canvas for live inference).
- **Manage experiments** via a model registry that persists configs and evaluation metrics alongside saved weights.

---

## Project Structure

```
bnn-app/
├── main.py                     # GUI entry point
├── requirements.txt
│
├── core/                       # Framework-agnostic ML pipeline
│   ├── config.py               # ExperimentConfig, TrainingConfig, InferenceConfig
│   ├── experiment.py           # ExperimentRunner orchestrator + CLI entry point
│   ├── registry.py             # ModelRegistry — save/load/list trained models
│   ├── gallery_registry.py     # Persistent gallery of drawn canvas samples
│   │
│   ├── data/
│   │   ├── datasets.py         # DataManager — train/val/test/OOD DataLoaders
│   │   ├── transforms.py       # Shared torchvision transform pipelines
│   │   ├── kmnist.py           # KMNIST dataset helper
│   │   └── notmnist.py         # NotMNIST dataset helper
│   │
│   ├── models/
│   │   └── alexnet.py          # AlexNetSmall — 5-block CNN for 28×28 inputs
│   │
│   ├── training/
│   │   └── trainer.py          # Trainer — epoch/batch loop with callbacks
│   │
│   ├── inference/
│   │   ├── base.py             # Predictor ABC
│   │   ├── deterministic.py    # DeterministicPredictor
│   │   └── mc_dropout.py       # MCDropoutPredictor (T stochastic forward passes)
│   │
│   ├── evaluation/
│   │   └── evaluator.py        # Evaluator — runs predictors over test + OOD sets
│   │
│   └── metrics/
│       └── metrics.py          # Accuracy, ECE, AUROC, entropy helpers
│
├── app/                        # PyQt6 GUI layer
│   ├── main_window.py          # Top-level QMainWindow with tab bar
│   ├── tabs/
│   │   ├── training_tab.py     # Config form, live plots, training controls
│   │   ├── testing_tab.py      # Model selector, evaluation results display
│   │   └── drawing_tab.py      # Freehand canvas + live inference panel
│   ├── widgets/
│   │   ├── canvas_widget.py    # Drawing canvas (mouse → PIL → tensor)
│   │   ├── gallery_widget.py   # Thumbnail grid of saved drawings
│   │   ├── live_plot_widget.py # Matplotlib-in-Qt live chart
│   │   └── model_selector_widget.py
│   └── workers/
│       ├── training_worker.py  # QThread wrapper around Trainer
│       ├── inference_worker.py # QThread wrapper around Evaluator
│       └── download_worker.py  # QThread for dataset downloads
│
├── assets/
│   └── style.qss               # Qt stylesheet (dark theme)
│
└── tests/
    ├── test_data.py            # DataManager splits, OOD transforms
    ├── test_inference.py       # Deterministic & MC Dropout predictor outputs
    ├── test_metrics.py         # Accuracy, ECE, AUROC calculations
    └── test_registry.py        # ModelRegistry save/load/list round-trips
```

Key design principle: `core/` has **no Qt dependency**. The training loop, evaluator, and predictors communicate via plain Python callbacks, which the GUI workers bridge to Qt signals. This means the entire ML pipeline is independently testable and scriptable.

---

## Configuration & Setup

### 1. Python virtual environment

Python **3.10 or later** is required.

```bash
# Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install CUDA (GPU support)

GPU acceleration is optional; the code falls back to CPU automatically. If you have an NVIDIA GPU and want faster training, install PyTorch with the matching CUDA toolkit **before** installing `requirements.txt` (so the CPU-only wheel is not pulled in first).

Check your driver's maximum supported CUDA version:

```bash
nvidia-smi
```

Then install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/). Example for **CUDA 12.1**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the appropriate tag for your CUDA version (e.g. `cu118` for CUDA 11.8). Skip this step entirely to use CPU only — PyTorch will be pulled in automatically with the rest of the requirements.

### 3. Install Python requirements

```bash
pip install -r requirements.txt
```

This installs PyQt6, Matplotlib, NumPy, SciPy, scikit-learn, Pillow, tqdm, and pytest. PyTorch/torchvision should already be present from step 2 (or will be installed as a CPU build if this is the first `pip install` in the environment).

---

## Running a Single Experiment (CLI)

`core/experiment.py` exposes a `__main__` block that trains a model and evaluates it without the GUI:

```bash
python -m core.experiment [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-name NAME` | `cli_run` | Label stored in the model registry |
| `--dataset DATASET` | `mnist` | Training dataset: `mnist`, `fashion_mnist`, `kmnist`, `not_mnist`, `omniglot` |
| `--epochs N` | `20` | Number of training epochs |
| `--lr FLOAT` | `0.001` | Adam learning rate |
| `--batch-size N` | `128` | Mini-batch size |
| `--dropout FLOAT` | `0.5` | Bernoulli dropout rate (used at inference too) |
| `--mc-samples N` | `50` | Number of stochastic forward passes for MC Dropout |
| `--train-size FLOAT` | `1.0` | Fraction of the training set to use (e.g. `0.1` for quick runs) |
| `--device DEVICE` | auto | `cuda` or `cpu`; defaults to CUDA if available |

**Examples:**

```bash
# Quick smoke-test: 3 epochs, 10 % of training data, CPU
python -m core.experiment --run-name smoke --epochs 3 --train-size 0.1 --device cpu

# Full Fashion-MNIST run on GPU with 100 MC samples
python -m core.experiment \
    --run-name fashion_full \
    --dataset fashion_mnist \
    --epochs 30 \
    --lr 5e-4 \
    --mc-samples 100

# KMNIST with a lower dropout rate
python -m core.experiment --run-name kmnist_low_drop --dataset kmnist --dropout 0.2
```

After training, results are printed to stdout and the model (weights + config + metrics) is saved under `saved_models/` so it can be loaded in the GUI later.

---

## Running the Test Suite

Tests live in `tests/` and use **pytest**. They mock dataset downloads so no internet connection is required.

```bash
# Run all tests
pytest tests/

# Run a specific test module
pytest tests/test_metrics.py

# Verbose output with short tracebacks
pytest tests/ -v --tb=short

# Stop on first failure
pytest tests/ -x
```

The four test modules cover:

- **`test_data.py`** — `DataManager` split sizes, OOD dataset registry, and shared transform pipelines (shape, dtype, value range).
- **`test_inference.py`** — output shapes and value constraints for `DeterministicPredictor` and `MCDropoutPredictor`.
- **`test_metrics.py`** — correctness of accuracy, ECE, AUROC, and entropy calculations.
- **`test_registry.py`** — `ModelRegistry` save / load / list round-trips and edge cases.

---

## Launching the GUI

```bash
python main.py
```

The application opens with three tabs:

- **Training** — configure and launch an experiment, watch live loss/accuracy curves, and stop early.
- **Testing** — select a saved model from the registry, run deterministic or MC Dropout evaluation, and inspect calibration and OOD detection metrics.
- **Drawing** — sketch a digit on the freehand canvas and get a live class prediction with uncertainty estimate from the selected model.

Datasets are downloaded automatically to a `datasets/` folder on first use. Trained models are saved to `saved_models/` and exports (plots, CSVs) go to `exports/`.