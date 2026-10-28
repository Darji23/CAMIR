# CAMIR Coding Plan: Four Phases (v2, 2026-10-27)

Build the data pipeline before the live router. Every policy, threshold and guarantee in the proposal is evaluated by replaying cached outputs, so the code that records them is what everything else sits on.

**What changed from v1:** Phases 2–4 now include verification-based escalation (policy P5, extension X1), the regime map (X2), adaptive recalibration (X3), a load study, an open-ended chat slice and a trace release. Phase 1 is unchanged except for a closure checklist.

## Priorities and scope-cut order

| Tier | Items |
|---|---|
| **Core (never cut)** | C1 escalation tax (P3, P4), X1 verification escalation (P5), X2 regime map, C3 conformal guarantee |
| **Supporting** | Load study (E12), open-ended chat slice (E13), adaptive recalibration (X3), trace release |
| **Higher risk** | C2 MoE signal, C4 agentic per-step routing |

**If time runs short, cut in this order:** C4 → C2 → X3 → E13 → E12. Do not cut the core.

## Decisions to lock before Phase 2 starts

| Decision | Why it matters |
|---|---|
| Medium tier: keep Qwen2.5-7B 4-bit | It is the fallback verifier for P5. Dropping it means P5 depends entirely on the hosted API. |
| Hosted provider and real prices | Needed for ρ, the skip rule and the regime map. |
| ε (2–5 points) and α | Define "matched quality" and the guarantee level. |
| Energy method (measured or labelled "estimated") | Determines whether joules can be claimed in results. |
| Max output length per dataset | Affects cost, timing and what the gate sees. |

## Phase 1: Skeleton and contracts (done; closure checklist)

**Goal:** one prompt runs through one tier and produces one complete trace row.

Built: repo, pinned environment, Makefile, trace schema (`TierResult`, `TraceRow`), `Tier` interface with local and hosted adapters, SQLite + Parquet store, 1,364-prompt corpus with 40/20/40 splits and a 20-prompt smoke set, spikes S1–S3.

**Close these before Phase 2 (about 2 days):**

1. Run `make test` and `make smoke`; save the output in `logs/`.
2. Run S2 once against the real hosted API with a spending cap (5–10 prompts, run twice; the second run must be free). Record real price, latency and whether the provider returns log-probs.
3. Label energy as **estimated** (constant 12.5 W × time) everywhere, or replace it with a measurement (for example `powermetrics`).
4. Add the medium tier adapter and run it on the smoke set.
5. Check that `config_hash` covers model, quantization, max tokens, temperature, prompt template and library versions.
6. Store the answer-token log-prob separately for multiple-choice prompts.
7. Add thermal state and cache-disabled flag to every timing row.
8. Add one accounting test: cost = tokens × price, checked against a hand-computed value.

## Phase 2: Trace-first data pipeline and baselines (295A, about weeks 7–14)

**Goal:** a cost–quality frontier produced entirely from cached data, plus every trace later phases will need, collected once.

### Spikes (do first, one day each)

- **S6, verifier access (new).** Can the hosted model return log-probs for text you supply (the small model's answer)? If not, can the local 7B score a 1.5B answer? The result decides who the P5 verifier is.
- **S4, MoE and agent feasibility (new, cheap).** Does a small open MoE model (for example OLMoE) run under MLX-LM with router logits readable? Does a lightweight multi-step tool-use benchmark run on the laptop? Doing this now means Phase 4 holds no surprises.

### Build

1. **Resumable trace collector.** Runs every tier on all prompts and caches by `(prompt_id, tier, config_hash)`. It skips finished rows and never re-pays for hosted calls.
2. **Verification traces (new).** For each prompt, also record the verifier's score of the small model's answer (mean log-prob under the verifier, prefill only). Store it with the cost and time of that pass. This is collected now because P5 can only be replayed if these scores exist.
3. **Service-time records (new).** Log per-tier service-time distributions for the later load study.
4. **Open-ended chat slice (new).** Choose about 200 prompts from a public chat dataset (check the license first) and run all tiers on them. Judging happens in Phase 3, when the policies exist.
5. **Scorers.** Option match for ARC and MMLU, final-answer match for GSM8K, sandboxed unit-test runner with timeouts and no network for HumanEval.
6. **Replay engine.** Takes a policy plus the traces and returns cost, quality, latency (mean and tail) and energy.
7. **Baselines, in this order.** Always-small, always-medium, always-large, random mix, oracle; then the log-prob cascade; then a **complexity-tier router modelled on LiteLLM** (re-implemented from its documentation; say so in the report); then the feature router.
8. **Timing rules, from the start.** `perf_counter`; load time separate; warm-up discarded; cache disabled; thermal state logged; dollars, seconds and joules never added.

### Exit

- **E1 table:** per-tier quality, cost, TTFT, TPOT, p50/p95/p99 latency, energy.
- **Replayed frontier plot** for the baselines.
- **S5:** a 50-prompt live run agrees with replay.
- Verification traces and the chat-slice traces are complete.

## Phase 3: Live router, tax-aware core, guarantees (295B, about weeks 1–10)

**Goal:** C1 with X1 and X2, and C3 with X3, working end to end. These are the core claims.

### Build

1. **Policy interface shared by live and replay.** The FastAPI service wraps the same `Policy` objects the replay engine uses. If live and replay share the policy code, fidelity problems point to real bugs.
2. **Accountant.** Every row labels useful work and wasted work (rejected small attempt, gate cost, verification cost). The wasted label is the escalation tax.
3. **P3 skip-rule** (try small first only if the predicted success probability exceeds ρ). Needs a success-probability estimator trained on the train split.
4. **P4 early-abort** (stop the small generation if the running mean log-prob falls below τ_abort after n0 tokens). Needs a mid-stream stop hook.
5. **P5 verification-based escalation (X1).** When the gate rejects the small answer, the verifier scores it in one prefill pass; keep the small answer if the score passes, otherwise the large model regenerates. Replay first (from the Phase 2 verification traces), live second.
6. **Conformal calibrator.** A small pure function: calibration traces in, threshold and guarantee out, for the loss ℓ = 1[accepted ∧ small wrong ∧ large right]. For P5, tie the verification threshold to λ so acceptance stays monotone. Unit-test on synthetic data where the true risk is known.
7. **Adaptive recalibration (X3).** Audit a random fraction of requests with the large tier to reveal the loss, and update the working risk level online. Report the audit cost.
8. **Regime map (X2).** For each (ρ, e) and latency target, compute the cheapest strategy at matched quality from measured Δ, W and latency; validate by replay. Output a figure and a small calculator script.
9. **Open-ended slice judging.** Blinded pairwise judging against the large model's answer, both presentation orders, a judge from a different model family, conflicts recorded as ties, judge self-disagreement reported.
10. **Minimal dashboard** and full hosted-tier integration (retries, rate limits, real prices).

### Experiments

| ID | What |
|---|---|
| E3 | Tax measurement (cost, seconds, tail, joules) |
| E4 | Policy frontiers: baselines, P2, P3, P4, P5 |
| E5 | Sensitivity to ρ |
| E8 | Guarantee coverage versus α, repeated splits, deliberate dataset shift, and adaptive recalibration with its audit cost |
| E10 | P5 against P2–P4: pass rate v_pass, c_V/c_L, wrong answers accepted by the verifier, net savings (P5 helps only when v_pass·c_L > c_V) |
| E11 | Regime map: predicted regions against replayed ones |
| E13 | Open-ended slice: gate quality and frontiers without ground truth |

### Exit

E3, E4, E8, E10 and E11 run from one command. **Decision gate at week 6:** if E10 shows P5 does not pay (verifier accepts too many wrong answers, or v_pass is too low), report it as a negative result and keep going with P3, P4 and C3.

## Phase 4: Load study, riskier contributions, hardening (295B, about weeks 8–14)

**Goal:** show the system under load, test C2 and C4, and make everything reproducible.

### Build

1. **Load study (E12).** A trace-driven discrete-event simulator of concurrent requests: measured service-time distributions per tier, a concurrency or rate cap on the hosted tier, one local stream on the M4. Report p95/p99 against arrival rate and how the tax shifts the tail. Validate with a small live run (several concurrent clients) on a subset.
2. **C2 (MoE signal), if S4 passed.** Compute normalized router entropy, its change across layers and expert-selection instability. Compare every signal (log-prob, margin, self-consistency, verification score, MoE statistics, hidden-state probe) on AUROC and calibration error using the same cached traces. If no MoE model runs, use the hidden-state probe and say so.
3. **C4 (agentic), if S4 passed.** A `StepController` over a multi-step tool-use set. Record per-step context length, because switching tiers re-sends the whole history. Compare always-small, always-large, a per-step threshold from C3 and a learned step router (E9). If no benchmark fits, use multi-step chains built from the same datasets.
4. **Trace release.** Publish the traces (with useful/wasted labels and verification scores) and the replay harness, after checking benchmark licenses and the hosted provider's terms on redistributing outputs.
5. **Hardening.** CI (calibrator and cost tests first), a container recipe, a fresh-clone rebuild test.

### Exit

E12 and (if built) E9 run; a one-command reproduction of every table and figure in the report; trace release ready or a written reason it cannot be released.

## Experiment index

| ID | Phase | Purpose |
|---|---|---|
| E1 | 2 | Per-tier characterization |
| E2 | 4 | Signal quality (AUROC, calibration error) |
| E3 | 3 | Tax measurement |
| E4 | 3 | Policy frontiers |
| E5 | 3 | Sensitivity to ρ |
| E6 | 3–4 | Tail latency |
| E7 | 2 | Replay versus live fidelity (S5) |
| E8 | 3 | Guarantee coverage, shift, adaptive recalibration |
| E9 | 4 | Agentic per-step policies |
| E10 | 3 | Verification-based escalation (P5) |
| E11 | 3 | Regime map |
| E12 | 4 | Behavior under load |
| E13 | 3 | Open-ended chat slice |

## Rules for every phase

- **Tests before trust:** test cost and timing accounting before anything is built on it.
- **Deterministic runs:** fixed seeds, greedy decoding, logged library versions.
- **Hosted calls:** few, capped, always cached. Do a 20-prompt run and check the bill before the full run.
- **Merge discipline:** each merge leaves `make smoke` passing.
- **Honest labels:** anything estimated (energy), mocked (S2) or re-implemented from documentation (the LiteLLM-style baseline) is labelled as such in tables and the report.
- **Negative results are results:** the tax may be small, P5 may not pay, the MoE signal may not help. Report them.

## First three days after Phase 1 closure

1. Lock the decisions table above as a team.
2. Run S6 (verifier access) and S4 (MoE and agent feasibility).
3. Start the collector on the smoke set including the verification score, then scale to the full corpus.
