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
