# CAMIR Coding Plan v3 (2026-10-28)

Companion to *CAMIR Project Proposal v3*. Same four-phase structure, now including: verification-based escalation (P5), the regime map, adaptive recalibration, the expanded and noisier datasets, hardware portability, and the pip package.

Build the data pipeline before the live router. Every policy, threshold and guarantee is evaluated by replaying cached outputs, so the code that records them is what everything else sits on.

## 1. Priorities and scope-cut order

| Tier | Items |
|---|---|
| **Core (never cut)** | C1 escalation tax (P3, P4), X1 verification escalation (P5), X2 regime map, C3 conformal guarantee, perturbed and harder datasets, portability layer, minimal pip package (library + CLI + mock mode) |
| **Supporting** | Load study (E12), open-ended chat slice (E13), adaptive recalibration (X3), second hardware profile (E15), OpenAI-compatible proxy, regime-map web page, trace release |
| **Higher risk** | C2 MoE signal, C4 agentic per-step routing |

**If time runs short, cut in this order:** C4 → C2 → second hardware profile → X3 → chat slice → load study → proxy and container extras.

## 2. Decisions to lock before Phase 2

| Decision | Why it matters |
|---|---|
| Keep the medium tier (Qwen2.5-7B 4-bit) | Fallback verifier for P5 |
| Hosted provider and real prices | Needed for ρ, skip rule, regime map, and dataset budget |
| ε (2–5 points) and α | Define matched quality and guarantee level |
| Energy method per profile (measured or labelled "estimated") | Determines whether joules can be claimed |
| Max output length per dataset | Affects cost, timing and the gate |
| Corpus size and spend cap | Each prompt costs a generation call and a verification pass; size from the real price |
| Judge model for the chat slice (different family from the large tier) | Needed for E13 |
| Second hardware profile (CPU-only first; NVIDIA if a teammate has the machine) | Needed for E15 |
| Package name (check PyPI) and license | Check model and dataset terms before choosing |

## 3. Datasets and data plan

**Why:** the core sets are clean and small. With n = 272 calibration prompts the conformal slack is about 1/(n+1) ≈ 0.4 points and the standard error of a 5% loss is about 1.3 points, so α below roughly 3% is not meaningful. Noisier, harder and larger sets fix both the realism and the guarantee.

| Source | Target size | Ground truth | Phase |
|---|---|---|---|
| ARC-Easy, MMLU, GSM8K, HumanEval | 1,364 (built) | option / number / unit tests | done |
| Perturbed twins (2–3 noise types over ~300 base prompts) | ~600–900 derived | inherited from the base prompt | 2 |
| Instruction-following set (IFEval-style) | ~500 | scripted checkers | 2 |
| Harder checkable sets (MATH-500 levels 4–5, MMLU-Pro, BigCodeBench subset) | ~300 | exact match / unit tests | 2 |
| Open-ended chat slice (WildChat candidate) | 200–300 | none; blinded pairwise judge | collect 2, judge 3 |
| Multi-step tool-use set (C4) | set by S4 | task success | 4 |

Total target is about 3,000 prompts, set by budget. Check sizes and licenses of every source before use.

### Build tasks (Phase 2)

1. **Perturbation generator** (`camir/datasets/perturb.py`). Deterministic, versioned, fixed seeds. Perturb **prose only**:
   - multiple choice: perturb the stem, leave options and letters intact;
   - GSM8K: do not touch numbers;
   - HumanEval: perturb docstring prose only; signature and tests untouched.
   Noise types: (i) character typos, swaps, dropped letters at a fixed rate; (ii) lowercase and stripped punctuation; (iii) irrelevant filler or context before the question; (iv, optional) a casual rewrite by a model with a spot audit.
2. **Invariant tests** for the generator: options unchanged, numbers unchanged, signature unchanged, same seed gives identical output.
3. **Group-aware splitting.** Give each prompt a `group_id`; a clean prompt and all its noisy twins share one. Split by `group_id` (stratified by source). Test: no `group_id` appears in two splits.
4. **Held-out source.** Keep one whole source (for example the instruction-following set or the chat slice) out of calibration, to be the dataset-shift test in E8.
5. **Scorers for the new sets.** IFEval-style checkers, math answer match for MATH-500, MMLU-Pro option match, BigCodeBench unit tests in the sandbox (timeouts, no network).
6. **Chat slice.** Choose and filter prompts, scrub personal data, store prompts and tier outputs. No raw chat logs in any release.
7. **Pilot first.** Run each new set on 20 prompts, check the bill, then launch the full run.
8. **Per-source reporting** in every table, not only pooled numbers.

### Notes
- The core sets may overlap the Qwen pretraining data. Rely on relative comparisons between policies and say so.
- Hypothesis to test (not assume): small models lose more accuracy under noise than large ones (E14).

## 4. Phase 1: Skeleton and contracts (done; closure checklist)

Built: repo, pinned environment, Makefile, trace schema (`TierResult`, `TraceRow`), `Tier` interface with local and hosted adapters, SQLite + Parquet store, 1,364-prompt corpus with splits and a 20-prompt smoke set, spikes S1–S3.

**Close these before Phase 2 (about 2–3 days):**

1. Run `make test` and `make smoke`; save output in `logs/`.
2. Run S2 against the real hosted API with a spending cap (5–10 prompts, twice; the second run must be free). Record real price, latency, and whether log-probs are returned.
3. Label energy as **estimated** everywhere, or measure it.
4. Add the medium-tier adapter and run it on the smoke set.
5. Make `config_hash` cover model, quantization, max tokens, temperature, prompt template, library versions **and `hardware_profile`**.
6. Store the answer-token log-prob separately for multiple-choice prompts.
7. Add thermal state and a cache-disabled flag to every timing row.
8. Add one accounting test: cost = tokens × price, checked by hand.
9. **Portability layer (new):**
   - add `hardware_profile` to `TraceRow`;
   - create `camir/backends/` with a backend interface (MLX-LM now; llama.cpp stub; vLLM optional) behind `Tier`;
   - make energy a per-profile plug-in (`powermetrics` on Apple, NVML on NVIDIA, RAPL where available, none on cloud).
10. **Package skeleton (new):** correct `pyproject.toml`, `camir/cli.py` stub (`calibrate`, `shadow`, `serve`, `--mock`), mock mode that needs no keys or models. Doing this now avoids a late refactor.

## 5. Phase 2: Trace-first data pipeline and baselines (295A, about weeks 7–14)

**Goal:** a cost–quality frontier from cached data, plus every trace later phases need, collected once.

### Spikes (one day each, first)
- **S6 verifier access:** can the hosted model return log-probs for text you supply (the small model's answer)? If not, can the local 7B score a 1.5B answer?
- **S4 MoE and agent feasibility:** does a small open MoE model (for example OLMoE) run under MLX-LM with router logits readable? Does a lightweight multi-step tool-use benchmark run on the laptop?

### Build
1. **Data build** (section 3 above): perturbation generator, new sets, group-aware splits, held-out source, scorers, chat slice.
2. **Resumable trace collector**, cached by `(prompt_id, tier, config_hash)`; skips finished rows; never re-pays for hosted calls.
3. **Verification traces:** for every prompt, the verifier's score of the small answer (prefill only), with its cost and time.
4. **Service-time records** per tier for the load study.
5. **Replay engine:** cost, quality, latency (mean and tail), energy.
6. **Baselines, in order:** always-small, always-medium, always-large, random mix, oracle; log-prob cascade; a complexity-tier router modelled on LiteLLM (re-implemented from its documentation; say so); feature router.
7. **Timing rules:** `perf_counter`; load time separate; warm-up discarded; cache disabled; thermal state logged; dollars, seconds and joules never added.

### Exit
- E1 table (per tier, per source): quality, cost, TTFT, TPOT, p50/p95/p99 latency, energy.
- Replayed frontier plot for the baselines.
- S5: a 50-prompt live run agrees with replay.
- Verification traces, expanded corpus and chat-slice traces complete.

## 6. Phase 3: Live router, tax-aware core, guarantees (295B, about weeks 1–10)

**Goal:** C1 with X1 and X2, C3 with X3, and noise robustness, end to end.

### Build
1. **Shared policy interface:** FastAPI service wraps the same `Policy` objects the replay engine uses.
2. **Accountant:** every row labels useful and wasted work (rejected small attempt, gate cost, verification cost). Wasted = escalation tax.
3. **P3 skip-rule** (try small first only if predicted success probability > ρ); estimator trained on the train split.
4. **P4 early-abort** (stop the small generation if running mean log-prob < τ_abort after n0 tokens); needs a mid-stream stop hook.
5. **P5 verification-based escalation (X1):** when the gate rejects the small answer, the verifier scores it in one prefill pass; keep it if it passes, otherwise the large model regenerates. Replay first, live second.
6. **Conformal calibrator** (pure function): calibration traces in, threshold and guarantee out, for ℓ = 1[accepted ∧ small wrong ∧ large right]. For P5 tie the verification threshold to λ so acceptance stays monotone. Unit-test on synthetic data with known risk.
7. **Adaptive recalibration (X3):** audit a random fraction of requests with the large tier; update the working risk level online; report audit cost.
8. **Regime map (X2):** for each (ρ, e) and latency target, the cheapest strategy at matched quality; validate by replay; figure plus calculator script.
9. **Chat-slice judging:** blinded pairwise vs the large model's answer, both orders, conflicts as ties, judge self-disagreement reported.
10. **Noise robustness (E14):** clean vs perturbed twins, per tier and noise type.
11. Minimal dashboard; full hosted-tier integration (retries, rate limits, real prices).

### Experiments
| ID | What |
|---|---|
| E3 | Tax measurement |
| E4 | Policy frontiers (baselines, P2–P5) |
| E5 | Sensitivity to ρ |
| E8 | Coverage vs α, repeated splits, held-out-source shift, adaptive recalibration with audit cost |
| E10 | P5 vs P2–P4: v_pass, c_V/c_L, wrong answers accepted by verifier, net savings (P5 helps only if v_pass·c_L > c_V) |
| E11 | Regime map: predicted vs replayed regions |
| E13 | Open-ended slice |
| E14 | Noise robustness |

### Exit
E3, E4, E8, E10, E11, E14 run from one command. **Week-6 gate:** if E10 shows P5 does not pay, report it as a negative result and continue with P3, P4 and C3.

## 7. Phase 4: Load, portability, riskier contributions, packaging (295B, about weeks 8–14)

**Goal:** system under load, one extra hardware profile, C2 and C4, and a pip-installable release.

### Build
1. **Load study (E12):** trace-driven discrete-event simulator with measured service-time distributions, a hosted-tier concurrency or rate cap, one local stream; p95/p99 vs arrival rate; validate with a small live run (several concurrent clients).
2. **Second hardware profile (E15):** re-collect local-tier traces on CPU-only (llama.cpp, same laptop) first; NVIDIA via a teammate's machine (WSL2 on Windows) if available; cloud only as a short capped session. Test whether regime-map boundaries move. Do not claim parity.
3. **C2 (MoE signal), if S4 passed:** router entropy, cross-layer change, expert-selection instability; compare all signals on AUROC and calibration error (E2). Fallback: hidden-state probe.
4. **C4 (agentic), if S4 passed:** `StepController`, per-step context length, E9. Fallback: chains built from the same datasets.
5. **Packaging (section 8), release, E16.**
6. **Trace release** (after checking licenses and the provider's terms on redistributing outputs; no raw chat logs).
7. **Hardening:** CI (calibrator and cost tests first), container recipe for CPU/NVIDIA backends, fresh-clone rebuild test.

### Exit
E12, E15 (if built), E9 (if built), E16; one-command reproduction of every table and figure; tagged release.

## 8. Packaging: pip library (`pip install camir`)

**Audience:** developers and small teams that already call an LLM API. A non-technical end user would need a desktop app (future work).

### Layout
```
camir/
  pyproject.toml  README.md  LICENSE
  camir/
    __init__.py
    cli.py                 # calibrate | shadow | serve | --mock
    backends/              # mlx_lm, llama_cpp, (vllm)
    tiers/ signals/ policies/ accounting/ traces/ datasets/ replay/ service/ eval/
  examples/camir.example.yaml
  tests/
```
Keep out of the package: `data/*.jsonl`, `traces/`, `logs/`, `.env`, model files, anything with prompts or keys.

### `pyproject.toml` (starting point)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "camir"            # check PyPI first; alternative: camir-router
version = "0.1.0"
description = "Cost-aware multi-model inference router with calibrated guarantees"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }   # choose deliberately after checking model/dataset terms
dependencies = ["pydantic", "fastapi", "uvicorn", "httpx", "pyarrow"]

[project.optional-dependencies]
mlx   = ["mlx-lm; sys_platform == 'darwin' and platform_machine == 'arm64'"]
llama = ["llama-cpp-python"]

[project.scripts]
camir = "camir.cli:main"
```

### Commands
| Command | What it does |
|---|---|
| `camir calibrate` | Takes a few hundred of the user's own prompts, runs the tiers, fits the threshold for a chosen α, writes the config file |
| `camir shadow` | Runs next to real traffic without routing; reports projected savings and quality loss |
| `camir serve` | OpenAI-compatible local proxy; routing on; falls back to the large tier on any error; optional audit sampling (X3) |
| `--mock` | No keys or models; used by the quickstart and the test suite |

Config file fields (example): tiers and prices, α, thresholds, fallback tier, audit rate, hardware profile.

### Extras (optional)
- Static regime-map web page (no install).
- Container image for CPU/NVIDIA backends (as far as we know Linux containers on a Mac cannot use Metal; check before promising).
- Plug-in for an existing gateway such as LiteLLM's custom routing hook (not yet tested).

### Release checklist
1. Check the package name on PyPI.
2. `uv build`; inspect with `unzip -l dist/*.whl` (code and example config only; no traces, prompts, keys, models).
3. Install the wheel in a clean virtual environment; `camir --help` and the `--mock` quickstart must work on macOS and Linux (**E16**).
4. Publish to **TestPyPI** first; install from it.
5. Publish to PyPI through trusted publishing from a tagged GitHub release (no stored token). PyPI versions cannot be re-uploaded; release deliberately.
6. Archive the release with a DOI for the thesis.
7. README states the limits plainly: the guarantee is marginal, assumes prompts similar to the calibration data, and was tested only on the listed benchmarks.

## 9. Experiment index

| ID | Phase | Purpose |
|---|---|---|
| E1 | 2 | Per-tier characterization (per source) |
| E2 | 4 | Signal quality (AUROC, calibration error) |
| E3 | 3 | Tax measurement |
| E4 | 3 | Policy frontiers |
| E5 | 3 | Sensitivity to ρ |
| E6 | 3–4 | Tail latency |
| E7 | 2 | Replay vs live fidelity (S5) |
| E8 | 3 | Guarantee coverage, shift, adaptive recalibration |
| E9 | 4 | Agentic per-step policies |
| E10 | 3 | Verification-based escalation (P5) |
| E11 | 3 | Regime map |
| E12 | 4 | Behavior under load |
| E13 | 3 | Open-ended chat slice |
| E14 | 3 | Noise robustness (clean vs perturbed twins) |
| E15 | 4 | Cross-environment regime map |
| E16 | 4 | Packaging check (clean install, mock quickstart, shadow mode) |

## 10. Rules for every phase

- **Tests before trust:** test cost and timing accounting before anything is built on it.
- **Deterministic runs:** fixed seeds, greedy decoding, logged library versions.
- **Hosted calls:** few, capped, always cached; pilot of 20 prompts and a bill check before any full run.
- **Merge discipline:** each merge leaves `make smoke` passing.
- **Honest labels:** anything estimated (energy), mocked (S2), synthetic (perturbed prompts) or re-implemented from documentation (the LiteLLM-style baseline) is labelled in tables and the report.
- **Per-source and per-profile reporting**, not only pooled numbers.
- **Negative results are results:** the tax may be small, P5 may not pay, the MoE signal may not help, the regime map may not move across hardware.

## 11. First three days after Phase 1 closure

1. Lock the decisions table (section 2) as a team.
2. Run S6 and S4.
3. Start the collector on the smoke set with verification scores; build the perturbation generator and its invariant tests; then scale to the full corpus.
