# CAMIR Coding Plan: Four Phases

Build the data pipeline before the live router. Every policy, threshold and guarantee in the proposal is evaluated by replaying cached outputs, so the code that records them is what everything else sits on.

## Phase 1: Skeleton and contracts (about weeks 1–6)

**Goal:** one prompt runs through one tier and produces one complete trace row.

- Create the repo, pinned environment and a `Makefile` or task runner. Do this first so all three of you run identical setups.
- Fix the **trace schema** before writing any model code. It is the contract everyone codes against.
- Define one `Tier` interface with local and hosted adapters behind it:

```
camir/
  tiers/      base.py (Tier.generate -> TierResult), mlx_local.py, hosted_api.py
  signals/    logprob.py, margin.py, consistency.py   (later: moe_router.py, probe.py)
  policies/   base.py, fixed.py, cascade.py            (later: skip_rule.py, early_abort.py)
  accounting/ cost.py, timing.py, energy.py
  traces/     schema.py, store.py (SQLite + Parquet)
  data/       loaders, scorers, sandbox/
  replay/     engine.py, metrics.py, stats.py
  service/    app.py (FastAPI)
  eval/       experiments E1..E9
  tests/
```

- `TierResult` should carry: text, token ids, per-token log-probs (or stats), prompt and output token counts, `t_prefill`, `t_ttft`, `t_decode`, and `energy_j`.
- Run the spikes here: log-probs from MLX-LM (S1), the hosted API (S2), energy (S3).
- **Exit:** a 20-prompt smoke set runs through the small and hosted tiers, and the trace rows validate.

## Phase 2: Trace-first data pipeline and baselines (about weeks 7–14)

**Goal:** a cost–quality frontier produced entirely from cached data.

- Build a resumable runner that executes every tier on every prompt and caches by `(prompt_id, tier, config_hash)`. It must skip finished rows and never re-pay for hosted calls.
- Write the scorers: option match for the multiple-choice sets, final-answer match for GSM8K, and a sandboxed runner with timeouts for HumanEval.
- Build the replay engine. It takes a policy plus the traces and returns cost, quality, latency and energy.
- Implement always-small, always-large, random and oracle first, then the log-prob cascade.
- Use these timing rules from the start:
  - Time with `perf_counter`, and keep model load time separate.
  - Warm up and discard the first runs.
  - Disable the cache.
  - Log thermal state.
  - Never add dollars, seconds and joules together.
- **Exit (proof-of-concept demo):** E1 per-tier table and a replayed frontier plot. A 50-prompt live run agrees with replay (S5).

## Phase 3: Live router and the tax-aware core (CMPE 295B, about weeks 1–10)

**Goal:** contributions C1 and C3, working end to end.

- Build the FastAPI service around the same `Policy` interface the replay engine uses. If the live path and replay share the policy code, fidelity problems are easy to find.
- Add the accountant so that wasted work (rejected small attempts and gate cost) is labelled in every trace row. That label is the escalation tax.
- Implement the skip-rule policy first and the early-abort policy second. Early abort needs a hook that stops generation mid-stream.
- Build the conformal calibrator as a small pure function: calibration traces in, threshold and guarantee out. Unit-test it on synthetic data where the true risk is known.
- Integrate the hosted tier fully and add a minimal dashboard.
- **Exit:** E3 (tax), E4 (frontiers) and E8 (coverage versus α) run from one command.

## Phase 4: Riskier contributions and hardening (CMPE 295B, about weeks 8–14)

**Goal:** C2 and C4, plus reproducibility.

- **C2 (MoE signal):** First check that a small MoE model runs under MLX-LM and that router logits can be read. If not, fall back to a hidden-state probe. Compare every signal on AUROC and calibration error using the same cached traces.
- **C4 (agentic):** Build a `StepController` over a multi-step tool-use set. Record per-step context length, because the re-send cost grows with it.
- **Hardening:** Add CI (the calibrator and cost tests first), a container recipe, a fresh-clone rebuild test and the dashboard.
- **Exit:** E9 plus a one-command reproduction of every table and figure.

## Rules for every phase

- **Tests before trust:** Test cost and timing accounting before anything is built on it, because a silent bug there invalidates every result.
- **Deterministic runs:** Use fixed seeds, greedy decoding and logged library versions.
- **Hosted calls:** Make only a few, and always cache them.
- **Merge discipline:** Each merge should leave `make smoke` passing.

## First three days

1. Repo, environment, and the trace schema agreed as a team.
2. `mlx_local.py` returning text and log-probs for one prompt. If MLX-LM can't expose log-probs cleanly, switch to the fallback runtime now.
3. `hosted_api.py` with caching, and one trace row written to SQLite.
