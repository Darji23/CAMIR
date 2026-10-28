.PHONY: setup test data spike1 spike2 spike3 smoke smoke-mock smoke-no-medium clean clean-results

PYTHON = .venv/bin/python3
PYTEST = .venv/bin/pytest
LOGS_DIR = logs
CHECKPOINTS_PHASE1 = checkpoints/phase_1

# 1. Environment Setup
setup:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Setting up Python 3.11 environment with uv..."
	uv venv .venv --python 3.11 | tee $(LOGS_DIR)/setup.log $(CHECKPOINTS_PHASE1)/console_logs/setup.txt
	uv pip install -e ".[dev]" | tee -a $(LOGS_DIR)/setup.log $(CHECKPOINTS_PHASE1)/console_logs/setup.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# 2. Run All Unit Tests
test:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Running unit test suite..."
	@$(PYTEST) tests/ -v | tee $(LOGS_DIR)/test.log $(CHECKPOINTS_PHASE1)/console_logs/test.txt

# 3. Download & Prepare Datasets
data:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Downloading and stratifying benchmark datasets (ARC-Easy, MMLU, GSM8K, HumanEval)..."
	@$(PYTHON) camir/data/download_and_prep.py | tee $(LOGS_DIR)/data.log $(CHECKPOINTS_PHASE1)/console_logs/data.txt

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

# 5. Phase 1 Smoke Pipeline (Exit Gate)
smoke:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Executing 20-Prompt Smoke Pipeline across Small (1.5B), Medium (7B), and Large (Hosted) tiers..."
	@$(PYTHON) scripts/run_smoke.py | tee $(LOGS_DIR)/smoke.log

smoke-mock:
	@mkdir -p $(LOGS_DIR) $(CHECKPOINTS_PHASE1)/console_logs $(CHECKPOINTS_PHASE1)/outputs
	@echo "Executing fast dry-run of 20-Prompt Smoke Pipeline (mock mode)..."
	@$(PYTHON) scripts/run_smoke.py --mock-local | tee $(LOGS_DIR)/smoke_mock.log

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
