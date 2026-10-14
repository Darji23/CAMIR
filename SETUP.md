# CAMIR Setup & Execution Guide (Phase 1)

This document provides step-by-step instructions for setting up the environment, running tests, executing feasibility spikes, and reproducing the Phase 1 smoke test pipeline on Apple Silicon (M-series).

---

## 1. Prerequisites

* **Hardware:** Apple Silicon Mac (M1/M2/M3/M4)
* **Operating System:** macOS Sonoma or later
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (recommended for fast, pinned builds) or standard `python3` / `pip`

If you don't have `uv` installed yet, you can install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Environment Setup

### Option A: Using `make` (Recommended)
From the root of the repository (`/Users/abhishekdarji/MyDrive/CAMIR`):

```bash
make setup
```

This will automatically create a `.venv` virtual environment using Python 3.11 and install all required dependencies (`mlx`, `mlx-lm`, `pydantic`, `datasets`, `pyarrow`, `openai`, `duckdb`, `rich`, `pytest`).

### Option B: Manual Setup with `uv`
```bash
# 1. Create a Python 3.11 virtual environment
uv venv .venv --python 3.11

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install CAMIR and development dependencies
uv pip install -e ".[dev]"
```

---

## 3. Benchmark Dataset Preparation

The 4 benchmark datasets (**ARC-Easy**, **MMLU**, **GSM8K**, and **HumanEval**) are already downloaded and stratified into:
* `data/smoke_20.jsonl` (20-prompt smoke test set)
* `data/train.jsonl` (545 prompts — 40%)
* `data/calibration.jsonl` (272 prompts — 20%)
* `data/test.jsonl` (547 prompts — 40%)
* `data/dataset_manifest.json`

If you ever need to re-download or regenerate the dataset splits from scratch, run:
```bash
make data
```
*(or `.venv/bin/python camir/data/download_and_prep.py`)*

---

## 4. Run Unit Tests

Execute the unit test suite to verify schema validation, storage operations, and accounting calculations:

```bash
make test
```
*(or `.venv/bin/pytest tests/ -v`)*

**Expected output:** 4 tests passing in `< 1s` across `test_traces.py`, `test_accounting.py`, and `test_tiers.py`.

---

## 5. Run Feasibility Spikes (Phase 1 Milestones)

Each spike de-risks a specific technical requirement in the project proposal:

### Spike S1: Local MLX-LM Log-probs & Top-2 Margins
Verifies loading `Qwen2.5-1.5B-Instruct-4bit` on Apple Silicon, step-by-step token generation, and extracting token-level log-probabilities and top-2 margins:
```bash
make spike1
```
*(or `.venv/bin/python spikes/s1_mlx_logprobs.py`)*  
*Note: The first time you run this, MLX-LM will download the small ~1GB 4-bit model weights.*

### Spike S2: Hosted API & Idempotent SQLite Caching
Verifies querying the large hosted tier and guarantees that repeated requests with the same `(prompt_id, tier, config_hash)` hit the local SQLite cache with **0ms latency** and **$0.00 duplicate spend**:
```bash
make spike2
```
*(or `.venv/bin/python spikes/s2_hosted_cache.py`)*

### Spike S3: Apple Silicon Energy Measurement (Joules)
Verifies that package energy is recorded in Joules without mixing units (Joules, seconds, and dollars are kept strictly separate):
```bash
make spike3
```
*(or `.venv/bin/python spikes/s3_m4_energy.py`)*

---

## 6. Phase 1 Smoke Pipeline (The Exit Gate)

The exit criterion for Phase 1 is running the **20-prompt smoke set** through both tiers and producing validated trace rows.

### Fast Dry-Run (Mock Mode)
Runs all 20 prompts through mock tiers in ~1 second to verify SQLite insertion, schema validation, and Parquet export:
```bash
make smoke-mock
```
*(or `.venv/bin/python scripts/run_smoke.py --mock-local`)*

### Full Live Run (Phase 1 Exit)
Runs all 20 prompts through the local MLX small model and the hosted large model:
```bash
make smoke
```
*(or `.venv/bin/python scripts/run_smoke.py`)*

---

## 7. Inspecting Output Traces

After running `make smoke` or `make smoke-mock`, outputs are saved to the `traces/` folder:

1. **SQLite Database:** `traces/camir_traces.sqlite`
   * Inspect with any SQLite viewer or the sqlite CLI:
     ```bash
     sqlite3 traces/camir_traces.sqlite "SELECT prompt_id, tier, cost_usd, t_total_s, energy_j FROM traces LIMIT 5;"
     ```

2. **Columnar Parquet File:** `traces/smoke_traces.parquet`
   * Used for downstream replay in Phase 2.
   * You can inspect it with Python:
     ```bash
     .venv/bin/python -c "import pandas as pd; df = pd.read_parquet('traces/smoke_traces.parquet'); print(df[['prompt_id', 'tier', 'cost_usd', 't_total_s']].head())"
     ```

---

## 8. Optional: Configuring a Live Hosted API Key

By default, the hosted large tier runs in **mock mode** so you can test without spending money or needing an API key.

If you wish to query a real remote provider (e.g. Together AI, OpenRouter, DeepInfra):
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   # Optional base URL (e.g. OpenRouter or Together):
   # OPENAI_BASE_URL=https://openrouter.ai/api/v1
   ```

---

## 9. Quick Command Reference

| Task | Make Command | Direct Python Command |
| :--- | :--- | :--- |
| **Setup environment** | `make setup` | `uv venv .venv --python 3.11 && uv pip install -e ".[dev]"` |
| **Run unit tests** | `make test` | `.venv/bin/pytest tests/ -v` |
| **Rebuild datasets** | `make data` | `.venv/bin/python camir/data/download_and_prep.py` |
| **Spike S1 (MLX Logprobs)** | `make spike1` | `.venv/bin/python spikes/s1_mlx_logprobs.py` |
| **Spike S2 (Hosted Cache)** | `make spike2` | `.venv/bin/python spikes/s2_hosted_cache.py` |
| **Spike S3 (M4 Energy)** | `make spike3` | `.venv/bin/python spikes/s3_m4_energy.py` |
| **Smoke Dry-Run** | `make smoke-mock` | `.venv/bin/python scripts/run_smoke.py --mock-local` |
| **Smoke Live Run** | `make smoke` | `.venv/bin/python scripts/run_smoke.py` |
| **Clean caches** | `make clean` | `rm -rf .pytest_cache/ traces/*.parquet` |
