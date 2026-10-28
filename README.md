# Cost-Aware Multi-Model Inference Router (CAMIR)

**Team:** Abhishek Darji, Aniket Anil Naik, Tamizh Selvan Manivannan  
**Institution:** San José State University — MS Software Engineering  
**Advisor:** Prof. Vijay Eranti  
**Project Repository:** `/Users/abhishekdarji/MyDrive/CAMIR`  

---

## 1. Project Overview & Research Context

**CAMIR** is a scientifically grounded, cost-aware dynamic inference router designed for local-plus-hosted hybrid deployments. It routes user queries dynamically between efficient on-device Small Language Models (SLMs) running locally on Apple Silicon (M-series) via Apple's MLX framework and large hosted models accessed through pay-per-token APIs.

### Core Research Contributions & Novelty
1. **Escalation Tax Accounting (C1):** In confidence-gated cascades, if the small model's completion fails verification, escalating to the large model throws away the small model's generation. We formalize and measure the *escalation tax* $W = c_S + g + c_L$ in terms of dollars, seconds, and Joules.
2. **Verification-Based Escalation (P5 / Extension X1):** Instead of making the large model regenerate the answer from scratch upon escalation, a verifier tier $V$ (the local medium 7B model or hosted model) reads the prompt and small model's answer in a single prefill forward pass to score it via mean log-probability $v(x, y_S)$.
3. **Calibrated Routing with Guarantees (C3):** A conformal risk calibrator producing bounds on quality loss relative to always calling the large model.
4. **Regime Map (Extension X2):** A decision-theoretic framework mapping cost ratios $\rho = c_L / c_S$ and escalation rates $e$ to identify whether cascading, pre-routing, or always-large is optimal.

---

## 2. Multi-Tier Model Architecture

CAMIR structures inference into three distinct tiers:

| Tier | Model Identifier | Execution Engine | Quantization / Format | Memory (RAM / VRAM) | Role in CAMIR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small Tier** | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Apple MLX-LM (Metal) | 4-bit Group-Quantized | ~880 MB | Default fast first-pass generator; provides token log-probs & top-2 margins |
| **Medium Tier** | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Apple MLX-LM (Metal) | 4-bit Group-Quantized | ~4.3 GB | Intermediate baseline and local fallback verifier for Policy P5 |
| **Large Tier** | `qwen/qwen-2.5-72b-instruct` | OpenAI-compatible REST API | FP16 Hosted (Remote) | N/A (Cloud hosted) | Gold-standard benchmark ceiling; provides high-accuracy fallback |

---

## 3. Model Configurations, Prompt Engineering & Finetuning Strategy

### 3.1 Finetuning Strategy: Are Foundation Models Finetuned in Phase 1?
> [!IMPORTANT]
> **No LLM weights are finetuned in Phase 1.**  
> CAMIR deliberately relies on frozen, off-the-shelf instruction-tuned models (`Qwen2.5-Instruct` family). Here is why:
> 1. **Scientific Benchmark Comparability:** Using standard frozen weights ensures that downstream evaluations on ARC-Easy, MMLU, GSM8K, and HumanEval accurately reflect published baselines.
> 2. **Deterministic Confidence Signals:** Log-probabilities and top-2 logit margins must reflect natural epistemic uncertainty rather than calibration artifacts induced by small-dataset fine-tuning.
> 3. **What IS Trained Later (Phase 3 Router Policy P3):** In Phase 3, CAMIR trains a lightweight **pre-generation skip-rule classifier** (e.g., Logistic Regression / MLP probe) on the 40% `train.jsonl` split. This classifier predicts whether the small model will succeed given the prompt text and embeddings. It trains the router gate—**never modifying the base LLM weights**.

### 3.2 Generation Parameters & Greedy Decoding
To eliminate non-deterministic sampling variance across all tiers, generation is locked to **greedy decoding**:
* `temperature = 0.0`: Completely deterministic output selection (argmax over logits).
* `seed = 42`: Fixed seed for MLX random number generators and hosted API providers.
* `max_tokens = 64` (smoke set) or task-specific maximums (e.g., 512 for code completion).
* `cache_disabled = True`: Prompt and KV caches are deliberately bypassed/cleared prior to latency measurements to ensure reported Time-To-First-Token (TTFT) and decode latencies are honest and cold-start isolated.

### 3.3 Prompt Template Standardization
Prompts are wrapped using standardized instruct templates matching each model's tokenizer:
* **Multiple Choice (ARC-Easy, MMLU):** Formatted with explicit option letters `(A), (B), (C), (D)` ending in `Answer:`. The router records the isolated log-probability of the choice token in `answer_token_logprob`.
* **Math Word Problems (GSM8K):** Multi-step chain-of-thought formatted to conclude with `#### <final_answer>`.
* **Code Completion (HumanEval):** Pure function signature and docstring with unit tests evaluated inside an isolated runner.

### 3.4 Reproducible Configuration Hashing (`config_hash`)
Every execution row is indexed by an immutable 16-character SHA-256 hash computed over 7 canonical parameters:
```python
normalized_config = {
    "model_id": "mlx-community/qwen2.5-1.5b-instruct-4bit",
    "quantization": "4bit",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42,
    "prompt_template": "chat_instruct",
    "library_versions": {"mlx": "0.22.0", "python": "3.11.13", "openai": "1.52.0"},
}
```
If any parameter changes (including library versions), the system computes a new hash, preventing corrupted trace replaying in Phase 2.

---

## 4. Environment & Hardware Prerequisites

* **Hardware:** Apple Silicon Mac (M1/M2/M3/M4). Verified on **Apple M4 (10 cores, 16.0 GB Unified Memory)**.
* **OS:** macOS Sonoma (14.0+) or macOS Sequoia (15.0+).
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (strongly recommended) or standard Python 3.11+.

### 4.1 Environment Installation
Install the pinned virtual environment with all development packages:

```bash
# Using Makefile (recommended)
make setup

# Or manually using uv:
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 4.2 Hosted API Configuration (Optional / Offline Default)
By default, CAMIR operates in **offline mock mode** for the large tier, allowing you to run all tests and smoke pipelines without internet access or paying API credits.

To run against a live hosted provider (OpenRouter, Together AI, Fireworks, DeepInfra):
1. Copy the environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your API credentials:
   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1  # or https://api.together.xyz/v1
   ```

---

## 5. Phase 1 Execution Guide (Step-by-Step)

Phase 1 establishes the skeleton contracts, storage engines, feasibility spikes, and benchmark datasets. Follow this order to execute Phase 1 from start to finish:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Phase 1 Execution Workflow                      │
├───────────────────┬───────────────────┬───────────────────┬────────────┤
│      Step 1       │      Step 2       │      Step 3       │   Step 4   │
│    Unit Tests     │   Dataset Prep    │    Spikes S1-S3   │ Smoke Gate │
│    `make test`    │    `make data`    │ `make spike1/2/3` │`make smoke`│
└───────────────────┴───────────────────┴───────────────────┴────────────┘
```

---

### Step 1: Run the Unit Test Suite
Verifies schema contracts, SQLite storage operations, hand-computed accounting math, and config hashing:

```bash
make test
```
* **Direct command:** `.venv/bin/pytest tests/ -v`
* **What it validates:**
  * `test_hand_computed_token_cost_accounting`: Exact match of token count $\times$ tariff against hand-calculated values.
  * `test_medium_tier_adapter_initialization`: Verifies `MediumMLXTier` defaults to `Qwen2.5-7B-Instruct-4bit`.
  * `test_config_hash_covers_all_fields`: Asserts that changing model, quantization, max tokens, temperature, prompt template, or package versions alters the hash.
  * `test_checkpoint_manager_lifecycle`: Confirms checkpoint txt files are created properly.
* **Checkpoint saved:** `checkpoints/phase_1/console_logs/test.txt`

---

### Step 2: Download & Stratify Benchmark Corpus
Downloads and partitions the 1,364-prompt benchmark dataset into standardized splits:

```bash
make data
```
* **Direct command:** `.venv/bin/python camir/data/download_and_prep.py`
* **Datasets sampled:**
  * **ARC-Easy:** 400 prompts (multiple choice science reasoning)
  * **MMLU:** 400 prompts across 57 humanities/STEM subjects
  * **GSM8K:** 400 prompts (multi-step math word problems)
  * **HumanEval:** 164 prompts (Python code generation)
* **Splits generated:**
  * `data/smoke_20.jsonl`: 20-prompt smoke set (5 from each benchmark)
  * `data/train.jsonl`: 545 prompts (40% for Phase 3 router training)
  * `data/calibration.jsonl`: 272 prompts (20% for C3 conformal calibration)
  * `data/test.jsonl`: 547 prompts (40% for out-of-sample evaluation)
* **Checkpoint saved:** `checkpoints/phase_1/console_logs/data.txt`

---

### Step 3: Execute Feasibility Spikes

#### Spike S1: Local MLX-LM Log-probs & Top-2 Margins
Verifies step-by-step token generation on Apple Silicon using `mlx-lm.generate.generate_step`, isolating model load time and extracting per-token logprobs and margins:
```bash
make spike1
```
* **Direct command:** `.venv/bin/python spikes/s1_mlx_logprobs.py`
* **Model downloaded:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~880 MB).
* **Expected metrics:** Decode speed ~45–55 tokens/s on M4; clear separation between confident tokens (margin $> 5.0$) and ambiguous tokens (margin $< 0.1$).
* **Checkpoints saved:**
  * `checkpoints/phase_1/console_logs/spike1.txt`
  * `checkpoints/phase_1/outputs/spike1_summary.txt`

#### Spike S2: Hosted API, Spend Cap & Idempotent Zero-Cost Caching
Executes queries against the large hosted tier (or mock fallback) with strict budget controls and tests that repeated queries hit the local SQLite cache:
```bash
make spike2
```
* **Direct command with options:**
  ```bash
  .venv/bin/python spikes/s2_hosted_cache.py --prompts 5 --spend-cap 0.05
  ```
* **What it validates:**
  1. **Spending Cap:** Halts immediately if cumulative cost exceeds `--spend-cap` (default \$0.05 USD).
  2. **Provider Log-probs:** Detects whether the remote provider returns logprobs for text tokens.
  3. **Zero-Spend Cache (Pass 2):** Runs the exact same queries a second time. Pass 2 must achieve **100% SQLite cache hits** with **< 5ms latency** and **$0.000000 additional cost**.
* **Checkpoints saved:**
  * `checkpoints/phase_1/console_logs/spike2.txt`
  * `checkpoints/phase_1/outputs/spike2_summary.txt`

#### Spike S3: Apple Silicon Energy Accounting in Joules
Verifies package energy tracking during local inference, enforcing strict dimensional separation (Joules, seconds, and dollars are never added together):
```bash
make spike3
```
* **Direct command:** `.venv/bin/python spikes/s3_m4_energy.py`
* **Energy model:** Explicitly labeled as **estimated** based on the calibrated M4 active decode profile ($12.5\text{ W} \times \text{time}$), with optional `powermetrics` fallback.
* **Checkpoints saved:**
  * `checkpoints/phase_1/console_logs/spike3.txt`
  * `checkpoints/phase_1/outputs/spike3_summary.txt`

---

### Step 4: Phase 1 Smoke Pipeline (The Exit Gate)

The exit criterion for Phase 1 is running the **20-prompt smoke set** through all three tiers, validating schemas, persisting rows to SQLite, and exporting to Parquet.

#### Option A: Fast Dry-Run (Mock Mode)
Tests the complete end-to-end pipeline in ~1 second without downloading large model weights:
```bash
make smoke-mock
```
* **Direct command:** `.venv/bin/python scripts/run_smoke.py --mock-local`

#### Option B: Live Multi-Tier Run (Small, Medium, and Large)
Executes all 20 prompts through Small (1.5B), Medium (7B), and Large (72B) tiers:
```bash
make smoke
```
* **Direct command:** `.venv/bin/python scripts/run_smoke.py`
* **Memory requirement:** ~6 GB unified memory to hold the 7B 4-bit model in memory.

#### Option C: Live Run (Small and Large only)
If running on a memory-constrained machine (e.g., 8 GB RAM) and you wish to skip downloading the 4.3 GB Medium 7B model:
```bash
make smoke-no-medium
```
* **Direct command:** `.venv/bin/python scripts/run_smoke.py --skip-medium`

#### Checkpoints saved from smoke pipeline:
* Full terminal console log: `checkpoints/phase_1/console_logs/smoke.txt`
* Structured metrics report: `checkpoints/phase_1/outputs/smoke_summary.txt`
* SQLite trace database: `traces/camir_traces.sqlite`
* Parquet export: `traces/smoke_traces.parquet`

---

## 6. Checkpoints & Output Reference

All phase executions automatically store structured `.txt` files in `checkpoints/` so you can review findings, console outputs, and hardware conditions at any point in the future:

```
checkpoints/
└── phase_1/
    ├── console_logs/
    │   ├── test.txt              # Complete pytest console output
    │   ├── data.txt              # Dataset preparation and split counts
    │   ├── spike1.txt            # Spike S1 token logs and timings
    │   ├── spike2.txt            # Spike S2 spend cap & cache hit tables
    │   ├── spike3.txt            # Spike S3 energy calculations
    │   └── smoke.txt             # Full smoke test table across tiers
    └── outputs/
        ├── spike1_summary.txt    # S1 summary: model load, decode speed, margins
        ├── spike2_summary.txt    # S2 summary: spend cap, log-probs, idempotency
        ├── spike3_summary.txt    # S3 summary: active watts, energy per burst
        └── smoke_summary.txt     # Exit gate validation report (row counts, costs)
```

---

## 7. Trace Inspection & Replay Verification

Once `make smoke` finishes, trace records are stored in both SQLite and Parquet:

### Querying the SQLite Database
```bash
# View sample recorded rows with confidence signals and accounting metrics
sqlite3 traces/camir_traces.sqlite \
  "SELECT prompt_id, tier, answer_token_logprob, cost_usd, t_total_s, energy_j, energy_method FROM traces LIMIT 6;"
```

### Inspecting Parquet in Python
```python
import pandas as pd

df = pd.read_parquet("traces/smoke_traces.parquet")
print(df[["prompt_id", "tier", "output_tokens", "cost_usd", "t_total_s", "energy_j"]].head())
```

---

## 8. Cleaning & Resetting for a Fresh Re-Run

To wipe all generated trace databases, parquet exports, old logs, and checkpoints for a completely clean re-run:

```bash
make clean-results
```

To clean Python bytecode and pytest caches:
```bash
make clean
```

---

## 9. Complete Command Reference

| Purpose | Make Target | Direct Command | Output Checkpoint |
| :--- | :--- | :--- | :--- |
| **Setup Virtualenv** | `make setup` | `uv venv .venv --python 3.11 && uv pip install -e ".[dev]"` | `checkpoints/phase_1/console_logs/setup.txt` |
| **Run Unit Tests** | `make test` | `.venv/bin/pytest tests/ -v` | `checkpoints/phase_1/console_logs/test.txt` |
| **Prepare Datasets** | `make data` | `.venv/bin/python camir/data/download_and_prep.py` | `checkpoints/phase_1/console_logs/data.txt` |
| **Spike S1 (MLX Logprobs)** | `make spike1` | `.venv/bin/python spikes/s1_mlx_logprobs.py` | `checkpoints/phase_1/outputs/spike1_summary.txt` |
| **Spike S2 (Hosted Cache)** | `make spike2` | `.venv/bin/python spikes/s2_hosted_cache.py --prompts 5` | `checkpoints/phase_1/outputs/spike2_summary.txt` |
| **Spike S3 (M4 Energy)** | `make spike3` | `.venv/bin/python spikes/s3_m4_energy.py` | `checkpoints/phase_1/outputs/spike3_summary.txt` |
| **Smoke Dry-Run (Fast)** | `make smoke-mock` | `.venv/bin/python scripts/run_smoke.py --mock-local` | `checkpoints/phase_1/outputs/smoke_summary.txt` |
| **Smoke Live (3 Tiers)** | `make smoke` | `.venv/bin/python scripts/run_smoke.py` | `traces/camir_traces.sqlite` & `checkpoints` |
| **Smoke (Small & Large)** | `make smoke-no-medium` | `.venv/bin/python scripts/run_smoke.py --skip-medium` | `traces/camir_traces.sqlite` |
| **Reset Earlier Results** | `make clean-results` | `rm -f traces/*.sqlite traces/*.parquet logs/*.log` | Wipes traces, logs & checkpoint files |
| **Clean Pytest Caches** | `make clean` | `rm -rf .pytest_cache/ htmlcov/` | Cleans temporary cache dirs |
