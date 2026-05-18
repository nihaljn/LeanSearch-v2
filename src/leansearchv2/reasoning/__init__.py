"""Reasoning mode: theorem -> sub-queries -> retrieve -> filter -> judge -> entry list.

Entry point: `run_reasoning(problem, retriever, llm, ...)`. No formal-sketch
generation is performed; the output is purely a ranked list of LeanSearch
entries together with the orchestration metadata (`status`, `big_loop_count`,
`steps`, etc.).
"""

from .run import Problem, ReasoningLLMs, ReasoningResult, run_reasoning

__all__ = ["Problem", "ReasoningLLMs", "ReasoningResult", "run_reasoning"]
