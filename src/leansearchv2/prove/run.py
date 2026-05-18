"""Prove-task orchestrator: simple reflection loop, one theorem -> verified proof.

Public entry point: `run_prove(problem, llm, retriever, verifier, ...)`.

Flow per problem:

    # 1. Optional pre-retrieval (only for retriever_mode='reasoning', which
    #    keys on the theorem statement; standard/none defer to reflection).
    if retriever_mode == 'reasoning':
        pre = await run_reasoning(problem, retriever, llm)
        initial_search = format(pre.entries)
    elif retriever_mode == 'standard_initial':
        queries = await get_init_queries(llm, problem)
        initial_search = format(await batch_search(retriever, queries))
    else:
        initial_search = ""

    # 2. Initial attempt.
    proof = await prover_init(llm, problem, initial_search)
    result = await verifier.verify(proof)
    if result.success: return SUCCESS

    # 3. Reflection rounds.
    for _ in range(reflection_rounds):
        if retriever_mode == 'standard':
            queries = await get_reflect_queries(llm, problem, proof, result.error_msg)
            reflect_search = format(await batch_search(retriever, queries))
        elif retriever_mode == 'reasoning':
            reflect_search = initial_search  # keep the pre-retrieval
        else:
            reflect_search = ""
        proof = await prover_reflect(llm, problem, proof, result.error_msg, reflect_search)
        result = await verifier.verify(proof)
        if result.success: return SUCCESS
    return FAIL

`retriever_mode` values:
- ``"none"`` — no retrieval at all (baseline (i) in Table 3).
- ``"standard"`` — query is generated from the prover's current attempt /
  error trace; fires only during reflection (matches paper's prose for
  the semantic search engines).
- ``"reasoning"`` — runs reasoning mode once on the theorem statement
  up front; same retrieval payload reused throughout reflection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from ..standard_client import SearchResult, StandardClient
from ..reasoning import ReasoningLLMs, run_reasoning
from ..reasoning.decompose import _parse_json
from ..reasoning.filter import _pretty_inline
from ..reasoning.run import Problem as ReasoningProblem
from . import prompts
from .verifier import Verifier, VerifyResult, extract_lean_code


log = logging.getLogger("leansearchv2.prove")

_DEFAULT_NUM_QUERIES = 5


@dataclass
class ProveLLMs:
    """LLM bundle for the prove task. `prover` handles initial/reflect proof
    generation; `query` generates reflect-time queries when retriever_mode
    is "standard"; `reasoning` is required when retriever_mode is
    "reasoning"."""
    prover: LLMClient
    query: LLMClient
    reasoning: ReasoningLLMs | None = None

    @classmethod
    def from_config(cls, *, with_reasoning: bool = False) -> "ProveLLMs":
        from ..config import get
        prover_name = str(get("PROVE_PROVER_LLM", "prove", "prover_llm", default="openai"))
        query_name = str(get("PROVE_QUERY_LLM", "prove", "query_llm", default=prover_name))
        cache: dict[str, LLMClient] = {}
        def _get(name: str) -> LLMClient:
            if name not in cache:
                cache[name] = LLMClient(name)
            return cache[name]
        return cls(
            prover=_get(prover_name),
            query=_get(query_name),
            reasoning=ReasoningLLMs.from_config() if with_reasoning else None,
        )


@dataclass
class ProveProblem:
    problem_id: str
    formal_statement: str
    """Full Lean source (with imports, namespaces, and a `:= by sorry` to fill in)."""
    informal_statement: str = ""
    header: str = ""
    """Optional prefix prepended to the prover's output when it omits `import` lines."""


@dataclass
class ProveAttempt:
    round: int
    proof: str
    success: bool
    error_msg: str = ""


@dataclass
class ProveResult:
    problem_id: str
    success: bool
    rounds_used: int
    final_proof: str
    final_error: str
    attempts: list[ProveAttempt] = field(default_factory=list)
    retriever_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "success": self.success,
            "rounds_used": self.rounds_used,
            "retriever_mode": self.retriever_mode,
            "final_proof": self.final_proof,
            "final_error": self.final_error,
            "attempts": [
                {"round": a.round, "success": a.success, "error_msg": a.error_msg}
                for a in self.attempts
            ],
        }


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "(No search results provided.)"
    return "\n\n".join(f"[{i}] {_pretty_inline(r)}" for i, r in enumerate(results))


async def _get_queries(
    llm: LLMClient,
    *,
    formal_statement: str,
    informal_statement: str,
    proof: str | None,
    error_msg: str | None,
    num_queries: int,
) -> list[str]:
    if proof is None:
        prompt = prompts.GET_QUERY_INIT.format(
            num_queries=num_queries,
            lean_code=formal_statement,
            informal_statement=informal_statement or "(no informal description)",
        )
    else:
        prompt = prompts.GET_QUERY_REFLECT.format(
            num_queries=num_queries,
            lean_code=formal_statement,
            informal_statement=informal_statement or "(no informal description)",
            proof=proof,
            error_msg=error_msg or "(no error reported)",
        )
    try:
        response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        data = _parse_json(response)
        queries = [str(q) for q in (data.get("queries") or [])]
        return queries[:num_queries]
    except Exception as e:
        log.warning(f"get_queries failed: {type(e).__name__}: {e}")
        return []


async def _batch_search(
    retriever: StandardClient,
    queries: list[str],
    top_k: int,
    concurrency: int,
) -> list[SearchResult]:
    if not queries:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str):
        async with sem:
            try:
                return await retriever.search(q, top_k=top_k)
            except Exception:
                return []

    nested = await asyncio.gather(*[one(q) for q in queries])
    seen: set[str] = set()
    out: list[SearchResult] = []
    for group in nested:
        for r in group:
            key = f"{'.'.join(r.result.module_name)}::{'.'.join(r.result.name)}"
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


async def _prove_call(
    llm: LLMClient,
    *,
    lean_code: str,
    search_results_str: str,
    failing_proof: str | None = None,
    error_msg: str | None = None,
) -> str:
    if failing_proof is None:
        prompt = prompts.PROVER_INIT.format(
            lean_code=lean_code, search_results=search_results_str,
        )
    else:
        prompt = prompts.PROVER_REFLECT.format(
            proof=failing_proof,
            error_msg=error_msg or "",
            search_results=search_results_str,
            lean_code=lean_code,
        )
    response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    code = extract_lean_code(response)
    return code or response


async def run_prove(
    problem: ProveProblem,
    llms: ProveLLMs | LLMClient,
    retriever: StandardClient | None,
    verifier: Verifier,
    *,
    retriever_mode: str = "standard",  # "none" | "standard" | "reasoning"
    reflection_rounds: int = 8,
    num_queries: int = _DEFAULT_NUM_QUERIES,
    search_top_k: int = 10,
    retriever_concurrency: int = 8,
    reasoning_search_top_k: int = 30,
    reasoning_big_loop: int = 3,
    reasoning_output_top_k: int = 30,
    verify_timeout_s: int = 600,
) -> ProveResult:
    if retriever_mode not in ("none", "standard", "reasoning"):
        raise ValueError(f"unknown retriever_mode: {retriever_mode}")
    if retriever_mode != "none" and retriever is None:
        raise ValueError(f"retriever_mode={retriever_mode!r} requires a StandardClient")
    if isinstance(llms, LLMClient):
        llms = ProveLLMs(prover=llms, query=llms, reasoning=ReasoningLLMs(sketch=llms, filter=llms, judge=llms))
    if retriever_mode == "reasoning" and llms.reasoning is None:
        raise ValueError("retriever_mode='reasoning' requires `llms.reasoning` to be set")

    # Pre-retrieval.
    initial_results: list[SearchResult] = []
    if retriever_mode == "reasoning":
        reasoning_problem = ReasoningProblem(
            problem_id=problem.problem_id,
            formal_statement=problem.formal_statement,
            informal_statement=problem.informal_statement,
        )
        try:
            pre = await run_reasoning(
                reasoning_problem,
                retriever,  # type: ignore[arg-type]
                llms.reasoning,
                search_top_k=reasoning_search_top_k,
                output_top_k=reasoning_output_top_k,
                big_loop=reasoning_big_loop,
            )
            initial_results = [r for _did, _score, r in pre.entries]
        except Exception as e:
            log.warning(f"[{problem.problem_id}] reasoning prefetch failed: {e}")

    initial_search_str = _format_search_results(initial_results)

    # Initial attempt.
    attempts: list[ProveAttempt] = []
    proof = await _prove_call(
        llms.prover, lean_code=problem.formal_statement, search_results_str=initial_search_str,
    )
    if proof and problem.header and not proof.strip().startswith("import"):
        proof = problem.header + "\n\n" + proof
    result = await verifier.verify(proof, timeout_s=verify_timeout_s)
    attempts.append(ProveAttempt(round=0, proof=proof, success=result.success, error_msg=result.error_msg))
    if result.success:
        return ProveResult(
            problem_id=problem.problem_id, success=True, rounds_used=0,
            final_proof=proof, final_error="", attempts=attempts,
            retriever_mode=retriever_mode,
        )

    # Reflection rounds.
    for r_idx in range(1, reflection_rounds + 1):
        if retriever_mode == "standard":
            queries = await _get_queries(
                llms.query,
                formal_statement=problem.formal_statement,
                informal_statement=problem.informal_statement,
                proof=proof, error_msg=result.error_msg,
                num_queries=num_queries,
            )
            search_results = await _batch_search(
                retriever, queries, top_k=search_top_k, concurrency=retriever_concurrency,  # type: ignore[arg-type]
            )
            search_str = _format_search_results(search_results)
        elif retriever_mode == "reasoning":
            search_str = initial_search_str
        else:
            search_str = "(No retrieval enabled.)"

        proof = await _prove_call(
            llms.prover, lean_code=problem.formal_statement, search_results_str=search_str,
            failing_proof=proof, error_msg=result.error_msg,
        )
        if proof and problem.header and not proof.strip().startswith("import"):
            proof = problem.header + "\n\n" + proof
        result = await verifier.verify(proof, timeout_s=verify_timeout_s)
        attempts.append(ProveAttempt(round=r_idx, proof=proof, success=result.success, error_msg=result.error_msg))
        if result.success:
            return ProveResult(
                problem_id=problem.problem_id, success=True, rounds_used=r_idx,
                final_proof=proof, final_error="", attempts=attempts,
                retriever_mode=retriever_mode,
            )

    return ProveResult(
        problem_id=problem.problem_id, success=False,
        rounds_used=reflection_rounds,
        final_proof=proof, final_error=result.error_msg,
        attempts=attempts, retriever_mode=retriever_mode,
    )
