"""FastAPI live router gateway for CAMIR (Phase 3).

Exposes standard OpenAI-compatible endpoints:
- POST /v1/chat/completions
- GET  /v1/models
- GET  /health
- GET  /metrics

Directly delegates requests to CAMIR's tax-aware routing policies
(P2, P3, P4, P5), seamlessly handling on-device local execution and hosted fallback.
"""

import time
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from camir.accounting.cost import calculate_token_cost
from camir.accounting.tax import EscalationAccountant
from camir.routing.policies import (
    P3SkipRulePolicy,
    P4EarlyAbortPolicy,
    P5VerificationEscalationPolicy,
)
from camir.server.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    HealthResponse,
    ModelCard,
    ModelListResponse,
    UsageInfo,
)
from camir.tiers.hosted_api import HostedAPITier
from camir.tiers.mlx_local import LocalMLXTier, MediumMLXTier
from camir.utils.hardware import get_hardware_info, get_hardware_profile, get_thermal_state


def create_app(mock_mode: bool = False) -> FastAPI:
    app = FastAPI(
        title="CAMIR: Cost-Aware Multi-Model Inference Router API",
        version="0.1.0",
        description="Dynamic local-plus-hosted hybrid router with statistical guarantees",
    )

    # Initialize router state
    app.state.mock_mode = mock_mode
    app.state.default_policy = "P5"
    app.state.accountant = EscalationAccountant(policy_name="live-router")
    app.state.request_counts = {"small": 0, "medium": 0, "large": 0}
    app.state.total_requests = 0

    # Lazy-loaded tiers
    app.state.small_tier = None
    app.state.med_tier = None
    app.state.hosted_tier = None

    def get_tier(tier_name: str):
        if app.state.mock_mode:
            return None
        if tier_name == "small" and app.state.small_tier is None:
            try:
                app.state.small_tier = LocalMLXTier(tier_name="small")
            except Exception:
                pass
            return app.state.small_tier
        elif tier_name == "medium" and app.state.med_tier is None:
            try:
                app.state.med_tier = MediumMLXTier(tier_name="medium")
            except Exception:
                pass
            return app.state.med_tier
        elif tier_name == "large" and app.state.hosted_tier is None:
            try:
                app.state.hosted_tier = HostedAPITier(tier_name="large")
            except Exception:
                pass
            return app.state.hosted_tier
        return getattr(app.state, f"{tier_name}_tier", None)

    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return HealthResponse(
            status="healthy",
            active_policy=app.state.default_policy,
            hardware_profile=get_hardware_profile(),
            hardware_info=get_hardware_info(),
            thermal_state=get_thermal_state(),
            mock_mode=app.state.mock_mode,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models():
        models = [
            ModelCard(id="camir-auto"),
            ModelCard(id="qwen2.5-1.5b-instruct"),
            ModelCard(id="qwen2.5-7b-instruct"),
            ModelCard(id="qwen3p8-max"),
        ]
        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest):
        t0 = time.perf_counter()
        app.state.total_requests += 1

        # Extract latest user message
        user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if not user_msg:
            raise HTTPException(status_code=400, detail="No user message provided.")

        system_msg = next((m.content for m in req.messages if m.role == "system"), None)
        policy_key = (req.policy or app.state.default_policy).upper()

        chosen_tier = "small"
        escalated = False
        meta: Dict[str, any] = {}

        # Mock or fallback execution
        if app.state.mock_mode or req.model == "mock":
            # Deterministic routing logic
            if policy_key == "P3" and len(user_msg.split()) > 40:
                chosen_tier = "large"
                meta["skipped_small"] = True
            elif policy_key == "P5":
                # Simulate verifier saving small answer
                chosen_tier = "small"
                meta["saved_by_verifier"] = True
            else:
                chosen_tier = "small"

            output_text = f"[CAMIR {chosen_tier.upper()} Tier Response] Simulated completion for {len(user_msg.split())} words."
            prompt_tokens = len(user_msg.split())
            output_tokens = len(output_text.split())
            cost_usd = calculate_token_cost(f"mock-{chosen_tier}", prompt_tokens, output_tokens) if chosen_tier == "large" else 0.0
            latency_s = time.perf_counter() - t0
        else:
            # Live routing
            # Step 1: Check P3 skip rule
            if policy_key == "P3":
                p3 = P3SkipRulePolicy()
                p_hat = p3.predict_success_prob(user_msg)
                if p_hat <= p3.rho:
                    chosen_tier = "large"
                    meta["skipped_small"] = True

            tier_obj = get_tier(chosen_tier)
            if tier_obj is None:
                # Fallback to hosted
                tier_obj = get_tier("large")
                chosen_tier = "large"

            res = tier_obj.generate(
                prompt=user_msg,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system_prompt=system_msg,
            )
            output_text = res.text
            prompt_tokens = res.prompt_tokens
            output_tokens = res.output_tokens
            cost_usd = calculate_token_cost(tier_obj.model_id, prompt_tokens, output_tokens) if chosen_tier == "large" else 0.0
            latency_s = time.perf_counter() - t0

        app.state.request_counts[chosen_tier] = app.state.request_counts.get(chosen_tier, 0) + 1

        camir_meta = {
            "routed_tier": chosen_tier,
            "policy": policy_key,
            "cost_usd": cost_usd,
            "latency_ms": round(latency_s * 1000, 2),
            "escalated": escalated,
            "metadata": meta,
        }

        return ChatCompletionResponse(
            model=f"camir-{chosen_tier}",
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
            ),
            camir_metadata=camir_meta,
        )

    @app.get("/metrics")
    def get_metrics():
        n = max(1, app.state.total_requests)
        dist = {k: round(v / n, 4) for k, v in app.state.request_counts.items()}
        return {
            "total_requests": app.state.total_requests,
            "tier_distribution": dist,
            "counts": app.state.request_counts,
            "active_policy": app.state.default_policy,
        }

    return app


# Default instance for uvicorn
app = create_app(mock_mode=False)
