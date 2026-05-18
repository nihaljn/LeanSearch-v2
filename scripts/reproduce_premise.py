"""Reproduce the LeanSearch v2 (reasoning) row of Table 2
(Global premise retrieval on MathlibMPR).

For each of the 69 problems in `benchmark/MathlibMPR.json`, run reasoning
mode against a running standard-mode endpoint, then score the returned
entries against the ground-truth premise groups.

Usage:
    python scripts/reproduce_premise.py [--url http://localhost:8000]
                                        [--output reports/premise]
                                        [--parallelism 4]
                                        [--smoke]

Output (default `reports/premise/`):
- per_query.jsonl   one record per problem (full reasoning trace + score)
- summary.json      aggregated Recall@k(group) + Covered@k
- prints the Table-2-style summary on stdout

Requires: LLM API key (env var named by `llm.api_key_env` in config.yaml,
default `OPENAI_API_KEY`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.asyncio import tqdm_asyncio

from leansearchv2 import StandardClient
from leansearchv2.config import get
from leansearchv2.eval.premise_metrics import KS, aggregate, score_one
from leansearchv2.reasoning import Problem, ReasoningLLMs, run_reasoning


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark" / "MathlibMPR.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "premise"

OUTPUT_TOP_K = 50  # paper reports k up to 50


log = logging.getLogger("reproduce_premise")


async def _run_one(
    row: dict,
    retriever: StandardClient,
    llms: ReasoningLLMs,
    *,
    search_top_k: int,
    big_loop: int,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        problem = Problem(
            problem_id=row["id"],
            formal_statement=row["formal_statement"],
            informal_statement=row.get("NL_main_result", ""),
            informal_proof="",
        )
        try:
            result = await run_reasoning(
                problem,
                retriever,
                llms,
                search_top_k=search_top_k,
                output_top_k=OUTPUT_TOP_K,
                big_loop=big_loop,
            )
        except Exception as e:
            log.warning(f"FAIL {row['id']}: {type(e).__name__}: {e}")
            return {
                "id": row["id"],
                "error": f"{type(e).__name__}: {e}",
                "retrieved": [],
                "scores": score_one([], row["premise_group"]),
            }
        retrieved_ids = [doc_id for doc_id, _score, _r in result.entries]
        scores = score_one(retrieved_ids, row["premise_group"])
        return {
            **result.to_dict(),
            "retrieved": retrieved_ids,
            "premise_group": row["premise_group"],
            "scores": {k: {"recall_group": scores["recall_group"][k], "covered": scores["covered"][k]} for k in KS},
        }


async def run(
    url: str,
    benchmark_path: Path,
    output_dir: Path,
    parallelism: int,
    search_top_k: int,
    big_loop: int,
    smoke: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    bench = json.loads(benchmark_path.read_text())
    if smoke:
        bench = bench[:3]
    log.info(f"benchmark: {len(bench)} problems")

    retriever = StandardClient(url=url)
    llms = ReasoningLLMs.from_config()
    sem = asyncio.Semaphore(parallelism)

    tasks = [
        asyncio.create_task(_run_one(
            row, retriever, llms,
            search_top_k=search_top_k, big_loop=big_loop, sem=sem,
        ))
        for row in bench
    ]
    per_query: list[dict] = []
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="reasoning"):
        per_query.append(await fut)

    valid_scores = [r["scores"] for r in per_query if "scores" in r]
    transposed = [
        {"recall_group": {k: r[k]["recall_group"] for k in KS}, "covered": {k: r[k]["covered"] for k in KS}}
        for r in valid_scores
    ]
    summary = aggregate(transposed, KS)

    (output_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_query) + "\n"
    )
    (output_dir / "summary.json").write_text(json.dumps({"n": len(per_query), "summary": summary}, indent=2))
    log.info(f"wrote {output_dir}/per_query.jsonl  &  summary.json")

    print()
    print(f"== Global premise retrieval, n={len(per_query)} ==")
    print(f"{'metric':<14}  " + "  ".join(f"@{k:<3}" for k in KS))
    for metric in ("recall_group", "covered"):
        cells = "  ".join(f"{summary[metric][k]:>5.1f}" for k in KS)
        print(f"{metric:<14}  {cells}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None,
                        help="Standard-mode endpoint (default: retriever.url from config)")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--search-top-k", type=int, default=None,
                        help="Per-subquery retrieve top-k (default: reasoning.search_top_k)")
    parser.add_argument("--big-loop", type=int, default=None,
                        help="Max judge re-plans (default: reasoning.big_loop)")
    parser.add_argument("--smoke", action="store_true", help="Run only the first 3 problems")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    url = args.url or get("RETRIEVER_URL", "retriever", "url", default="http://localhost:8000")
    search_top_k = args.search_top_k if args.search_top_k is not None else int(
        get("REASONING_SEARCH_TOP_K", "reasoning", "search_top_k", default=30)
    )
    big_loop = args.big_loop if args.big_loop is not None else int(
        get("REASONING_BIG_LOOP", "reasoning", "big_loop", default=3)
    )
    return asyncio.run(run(
        url=url,
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output),
        parallelism=args.parallelism,
        search_top_k=search_top_k,
        big_loop=big_loop,
        smoke=args.smoke,
    ))


if __name__ == "__main__":
    sys.exit(main())
