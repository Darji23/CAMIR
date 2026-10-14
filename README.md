# Cost-Aware Multi-Model Inference Router (CAMIR)

**Team:** Abhishek Darji, Aniket Anil Naik, Tamizh Selvan Manivannan  
**Institution:** San José State University — MS Software Engineering  
**Advisor:** Prof. Vijay Eranti  

CAMIR is a cost-aware multi-model inference router designed to dynamically route queries between local on-device SLMs (e.g., Qwen2.5-1.5B / 7B running on Apple Silicon M4 via MLX) and large hosted models (e.g., Qwen2.5-72B via API).

## Quick Start
See [SETUP.md](file:///Users/abhishekdarji/MyDrive/CAMIR/SETUP.md) for full installation and execution instructions.

```bash
make test         # Run unit test suite
make spike1       # Run Spike S1 (MLX log-probs and top-2 margins)
make spike2       # Run Spike S2 (Hosted API + zero-cost caching)
make spike3       # Run Spike S3 (Apple Silicon energy in Joules)
make smoke-mock   # Fast dry-run of the 20-prompt smoke pipeline
make smoke        # Full live smoke test execution (Phase 1 Exit Gate)
```

## Phase 1 Deliverables
- [x] Standardized environment (`uv`, Python 3.11, `pyproject.toml`, `Makefile`)
- [x] Benchmark corpus downloaded & stratified (ARC-Easy, MMLU, GSM8K, HumanEval in `data/`)
- [x] Trace schema (`camir/traces/schema.py`) and dual storage SQLite + Parquet (`camir/traces/store.py`)
- [x] Accounting system for latency, token costs, and Joules (`camir/accounting/`)
- [x] Feasibility spikes S1, S2, and S3 (`spikes/`)
- [x] 20-prompt smoke test pipeline (`scripts/run_smoke.py`)
