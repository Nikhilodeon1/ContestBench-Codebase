"""Model sweep registry.

Primary analysis (novelty claim 3): gpt-oss-20b and gpt-oss-120b crossed with
reasoning_effort. Groq-hosted robustness-panel models are declared but commented
out until needed for the full run.
"""

from __future__ import annotations

GPT_OSS_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
REASONING_EFFORTS = ["low", "medium", "high"]


def sweep_specs(temperature: float = 0.0, efforts: list[str] | None = None) -> list[dict]:
    efforts = efforts or REASONING_EFFORTS
    specs = []
    for model in GPT_OSS_MODELS:
        short = model.split("/")[-1]
        for effort in efforts:
            specs.append({
                "label": f"{short}:{effort}",
                "model": model,
                "params": {"reasoning_effort": effort, "temperature": temperature},
            })
    return specs


# Robustness panel for the full run (Groq-hosted). deepseek-r1-distill is a
# distill of R1, not full R1 -- flagged as a limitation in the spec.
ROBUSTNESS_PANEL = [
    {"label": "deepseek-r1-distill", "model": "deepseek-r1-distill-llama-70b",
     "params": {"temperature": 0.0}},
    {"label": "llama-3.3-70b", "model": "llama-3.3-70b-versatile",
     "params": {"temperature": 0.0}},
]

# Primary panel (Fix 3): Claude family, capability x reasoning. Same lab/training
# pipeline so capability isn't confounded with cross-vendor calibration choices.
# capability_rank: Haiku < Sonnet < Opus (release-generation/size ordering).
CLAUDE_MODELS = [
    ("haiku", "claude-haiku-4-5-20251001", 1),
    ("sonnet", "claude-sonnet-5", 2),
    ("opus", "claude-opus-4-8", 3),
]


def claude_panel(thinking_budget: int = 2000) -> list[dict]:
    specs = []
    for short, model, rank in CLAUDE_MODELS:
        specs.append({"label": f"{short}:standard", "model": model,
                      "capability_rank": rank,
                      "params": {"thinking": False, "temperature": 0.0,
                                 "reasoning_effort": "standard"}})
        specs.append({"label": f"{short}:thinking", "model": model,
                      "capability_rank": rank,
                      "params": {"thinking": True, "thinking_budget": thinking_budget,
                                 "temperature": 1.0, "reasoning_effort": "thinking"}})
    return specs
