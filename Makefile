.PHONY: setup test data data-full spike1 spike2 spike3 spike4 spike6 traces phase2 phase2-data phase2-replay phase3 phase4 serve calibrate regime-map smoke smoke-no-medium clean clean-results

PYTHON = .venv/bin/python3
PYTEST = .venv/bin/pytest
LOGS_DIR = logs
CHECKPOINTS_PHASE1 = checkpoints/phase_1
CHECKPOINTS_PHASE2 = checkpoints/phase_2

# 1. Environment Setup
setup:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Setting up Python 3.11 environment with uv..."
	uv venv .venv --python 3.11 | tee $(LOGS_DIR)/setup.log $(CHECKPOINTS_PHASE1)/console_logs/setup.txt
	uv pip install -e ".[dev]" | tee -a $(LOGS_DIR)/setup.log $(CHECKPOINTS_PHASE1)/console_logs/setup.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# 2. Run All Unit Tests
test:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Running unit test suite..."
	@$(PYTEST) tests/ -v | tee $(LOGS_DIR)/test.log $(CHECKPOINTS_PHASE1)/console_logs/test.txt

# 3. Download & Prepare Datasets (all 11 sources from CAMIR_Project_Proposal_v3)
data:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Downloading and stratifying ALL benchmark datasets (10 sources + perturbed twins)..."
	@$(PYTHON) camir/data/download_and_prep.py | tee $(LOGS_DIR)/data.log $(CHECKPOINTS_PHASE1)/console_logs/data.txt

data-full: data
	@echo "Building Phase 2 group-aware augmented corpus from all 11 sources..."
	@$(PYTHON) scripts/prepare_phase2_data.py

# 4. Feasibility Spikes
spike1:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Running Spike S1 (MLX-LM Log-probs and Margins on Apple Silicon)..."
	@$(PYTHON) spikes/s1_mlx_logprobs.py | tee $(LOGS_DIR)/spike1.log

spike2:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Running Spike S2 (Hosted API, Spend Cap, and Idempotent SQLite Caching)..."
	@$(PYTHON) spikes/s2_hosted_cache.py | tee $(LOGS_DIR)/spike2.log

spike3:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Running Spike S3 (Apple Silicon Energy Measurement in Joules)..."
	@$(PYTHON) spikes/s3_m4_energy.py | tee $(LOGS_DIR)/spike3.log

spike4:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Running Spike S4 (MoE and Multi-Step Agent Feasibility)..."
	@$(PYTHON) spikes/s4_moe_agent_feasibility.py | tee $(LOGS_DIR)/spike4.log

spike6:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Running Spike S6 (Verifier Access & Prefill Scoring)..."
	@$(PYTHON) spikes/s6_verifier_access.py | tee $(LOGS_DIR)/spike6.log

# 5. Phase 2 Pipelines
phase2-data:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Building Phase 2 augmented corpus with group-aware splitting and held-out source..."
	@$(PYTHON) scripts/prepare_phase2_data.py | tee $(LOGS_DIR)/phase2_data.log

traces:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Collecting real inference traces across Small, Medium, and Large tiers..."
	@$(PYTHON) scripts/collect_traces.py --per-dataset 5 | tee $(LOGS_DIR)/collect_traces.log

phase2-replay:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE2)/console_logs $(CHECKPOINTS_PHASE2)/outputs
	@echo "Running Phase 2 trace replay, E1 table, and baseline frontiers..."
	@$(PYTHON) scripts/run_phase2.py | tee $(LOGS_DIR)/phase2_replay.log

phase2: phase2-data phase2-replay

# 6. Phase 3 Pipelines & Server
CHECKPOINTS_PHASE3 = checkpoints/phase_3

phase3:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE3)/console_logs $(CHECKPOINTS_PHASE3)/outputs
	@echo "Running Phase 3 Master Pipeline (P3, P4, P5, Conformal Risk, Adaptive Drift, Regime Map)..."
	@$(PYTHON) scripts/run_phase3.py | tee $(LOGS_DIR)/phase3.log

# 7. Phase 4 Pipelines: Load, Portability, MoE Signals & Agentic Routing
CHECKPOINTS_PHASE4 = checkpoints/phase_4

phase4:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE4)/console_logs $(CHECKPOINTS_PHASE4)/outputs
	@echo "Running Phase 4 Master Pipeline (E12 Load Study, E15 Hardware Shift, C2 MoE Signals, C4 Agentic Routing)..."
	@$(PYTHON) scripts/run_phase4.py | tee $(LOGS_DIR)/phase4.log

serve:
	@echo "Starting CAMIR live router gateway server on http://127.0.0.1:8000..."
	@$(PYTHON) -m camir.cli serve --host 127.0.0.1 --port 8000

calibrate:
	@echo "Calibrating conformal risk threshold..."
	@$(PYTHON) -m camir.cli calibrate

regime-map:
	@echo "Computing regime map optimal policy boundaries..."
	@$(PYTHON) -m camir.cli regime-map --rho 20.0 --e 0.35


# 5. Phase 1 Smoke Pipeline (Exit Gate)
smoke:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Executing 20-Prompt Smoke Pipeline across Small (1.5B), Medium (7B), and Large (Hosted) tiers..."
	@$(PYTHON) scripts/run_smoke.py | tee $(LOGS_DIR)/smoke.log

smoke-no-medium:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Executing 20-Prompt Smoke Pipeline (Small 1.5B and Large tiers)..."
	@$(PYTHON) scripts/run_smoke.py --skip-medium | tee $(LOGS_DIR)/smoke_no_medium.log

# 6. Delete Earlier Execution Traces & Results (Clean Slate for Re-run)
clean-results:
	@echo "Deleting all generated trace databases, parquet exports, and old execution logs..."
	rm -f traces/*.sqlite traces/*.parquet traces/*.sqlite-journal
	rm -f $(LOGS_DIR)/*.log
	rm -rf $(CHECKPOINTS_PHASE1)/console_logs/* $(CHECKPOINTS_PHASE1)/outputs/*
	@echo "Execution results deleted. Ready for fresh re-run."

# 7. Clean Temporary Files
clean:
	rm -rf .pytest_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
