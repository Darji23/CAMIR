.PHONY: setup test data spike1 spike2 spike3 smoke smoke-mock clean

PYTHON = .venv/bin/python3
PYTEST = .venv/bin/pytest
LOGS_DIR = logs

# 1. Environment Setup
setup:
	@mkdir -p $(LOGS_DIR)
	@echo "Setting up Python 3.11 environment with uv..."
	uv venv .venv --python 3.11 | tee $(LOGS_DIR)/setup.log
	uv pip install -e ".[dev]" | tee -a $(LOGS_DIR)/setup.log
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# 2. Run All Unit Tests
test:
	@mkdir -p $(LOGS_DIR)
	@echo "Running unit test suite..."
	@$(PYTEST) tests/ -v | tee $(LOGS_DIR)/test.log

# 3. Download & Prepare Datasets
data:
	@mkdir -p $(LOGS_DIR)
	@echo "Downloading and stratifying benchmark datasets (ARC-Easy, MMLU, GSM8K, HumanEval)..."
	@$(PYTHON) camir/data/download_and_prep.py | tee $(LOGS_DIR)/data.log

# 4. Feasibility Spikes
spike1:
	@mkdir -p $(LOGS_DIR)
	@echo "Running Spike S1 (MLX-LM Log-probs and Margins on Apple Silicon)..."
	@$(PYTHON) spikes/s1_mlx_logprobs.py | tee $(LOGS_DIR)/spike1.log

spike2:
	@mkdir -p $(LOGS_DIR)
	@echo "Running Spike S2 (Hosted API & Idempotent SQLite Caching)..."
	@$(PYTHON) spikes/s2_hosted_cache.py | tee $(LOGS_DIR)/spike2.log

spike3:
	@mkdir -p $(LOGS_DIR)
	@echo "Running Spike S3 (Apple Silicon Energy Measurement in Joules)..."
	@$(PYTHON) spikes/s3_m4_energy.py | tee $(LOGS_DIR)/spike3.log

# 5. Phase 1 Smoke Pipeline (Exit Gate)
smoke:
	@mkdir -p $(LOGS_DIR)
	@echo "Executing 20-Prompt Smoke Pipeline across Small (MLX) and Large (Hosted) tiers..."
	@$(PYTHON) scripts/run_smoke.py | tee $(LOGS_DIR)/smoke.log

smoke-mock:
	@mkdir -p $(LOGS_DIR)
	@echo "Executing fast dry-run of 20-Prompt Smoke Pipeline (mock mode)..."
	@$(PYTHON) scripts/run_smoke.py --mock-local | tee $(LOGS_DIR)/smoke_mock.log

# 6. Clean Temporary Files
clean:
	rm -rf .pytest_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
