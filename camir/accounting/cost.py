"""Cost accounting for hosted API and local inference tiers.

Calculates token-based dollar costs based on published tariffs.
Local tiers are recorded as $0.00 token cost.
"""

from typing import Dict, Tuple

# Pricing in USD per 1 Million tokens: (input_price_per_1M, output_price_per_1M)
MODEL_PRICING_CATALOG: Dict[str, Tuple[float, float]] = {
    # Hosted Large Tier - Fireworks AI (current)
    "accounts/fireworks/models/qwen3p8-max": (0.90, 0.90),
    "qwen3p8-max": (0.90, 0.90),
    # Hosted Large Tier - Fireworks AI (deprecated - no longer available)
    "accounts/fireworks/models/qwen2p5-72b-instruct": (0.90, 0.90),
    "fireworks/qwen2p5-72b-instruct": (0.90, 0.90),
    "qwen2p5-72b-instruct": (0.90, 0.90),
    "accounts/fireworks/models/qwen-2.5-72b-instruct": (0.90, 0.90),
    # Hosted Large Tier - OpenRouter / Together AI
    "qwen/qwen-2.5-72b-instruct": (0.35, 0.40),
    "qwen2.5-72b-instruct": (0.35, 0.40),
    "togethercomputer/qwen2.5-72b-instruct": (0.35, 0.40),
    "meta-llama/llama-3.3-70b-instruct": (0.40, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    # Local Small & Medium Tiers (Zero API token cost)
    "local": (0.0, 0.0),
    "mlx-community/qwen2.5-1.5b-instruct-4bit": (0.0, 0.0),
    "mlx-community/qwen2.5-7b-instruct-4bit": (0.0, 0.0),
}


def calculate_token_cost(model_id: str, prompt_tokens: int, output_tokens: int) -> float:
    """Calculate token financial cost in USD.
    
    Formula: (prompt_tokens / 1e6 * input_price) + (output_tokens / 1e6 * output_price)
    """
    key = model_id.lower().strip()
    if key.startswith("mlx-") or "local" in key:
        return 0.0

    pricing = MODEL_PRICING_CATALOG.get(key)
    if not pricing:
        # Check partial match
        for catalog_key, catalog_pricing in MODEL_PRICING_CATALOG.items():
            if catalog_key in key or key in catalog_key:
                pricing = catalog_pricing
                break

    if not pricing:
        # Default conservative hosted rate: $0.35 / $0.40 per 1M
        pricing = (0.35, 0.40)

    input_price, output_price = pricing
    cost = (prompt_tokens / 1_000_000.0 * input_price) + (output_tokens / 1_000_000.0 * output_price)
    return round(cost, 8)
