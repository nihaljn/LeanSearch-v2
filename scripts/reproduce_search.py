"""Reproduce the LeanSearch v2 rows of Table 1 (Search task on MathlibQR).

Runs two retriever configurations against `benchmark/MathlibQR.json` by
querying a running standard-mode endpoint (start it with `./scripts/serve.sh`
on a GPU pod first):

  - retriever-only:  embed -> cuVS top-k         (rerank=False)
  - rerank:          embed -> cuVS top-200 -> rerank top-k  (rerank=True)

Outputs (default `reports/search/`):
  - per_query.jsonl   one record per (query, config)
  - summary.json      slice × config × k -> ndcg/recall
  - prints the Table-1-style summary on stdout

Slices reported:
  - fair (810 rows / 171 declarations): shared across the four systems in the
    paper, matching Table 1.
  - full (946 rows): all queries; rows whose ground-truth declaration is
    absent from the served corpus simply score 0.

The 810-row fair subset is defined by `benchmark/MathlibQR_shared171.json`.

Usage:
    python scripts/reproduce_search.py [--url http://localhost:8000]
                                       [--output reports/search]
                                       [--parallelism 4]
                                       [--smoke]
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
from leansearchv2.eval.search_metrics import ndcg_at_k, recall_at_k, summarize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark" / "MathlibQR.json"
DEFAULT_SHARED = REPO_ROOT / "benchmark" / "MathlibQR_shared171.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "search"

QUERY_FIELDS = ["q1a_lean", "q1b_latex", "q1c_natural", "q2_slogan", "q3_nickname", "q4_special_case"]
K_VALUES = [1, 5, 10, 50, 100]
TOP_K = 100
RETRIEVE_K_RERANK = 200

CONFIGS = ["retriever_only", "rerank"]


log = logging.getLogger("reproduce_search")


def _flatten(bench: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in bench:
        for qf in QUERY_FIELDS:
            q = (entry.get(qf) or "").strip()
            if not q:
                continue
            rows.append({
                "id": f"{entry['id']}_{qf}",
                "original_id": entry["id"],
                "query_type": qf,
                "full_name": entry["full_name"],
                "difficulty": entry.get("difficulty", ""),
                "kind": entry.get("kind", ""),
                "query": q,
            })
    return rows


async def _run_one(
    client: StandardClient, row: dict, mode: str, sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        rerank = mode == "rerank"
        retrieve_k = RETRIEVE_K_RERANK if rerank else TOP_K
        try:
            results = await client.search(row["query"], top_k=TOP_K, rerank=rerank, retrieve_k=retrieve_k)
        except Exception as e:
            log.warning(f"FAIL {row['id']} mode={mode}: {type(e).__name__}: {e}")
            results = []
    retrieved = [".".join(r.result.name) for r in results]
    gt = row["full_name"]
    return {
        "id": row["id"],
        "original_id": row["original_id"],
        "query_type": row["query_type"],
        "full_name": gt,
        "difficulty": row["difficulty"],
        "kind": row["kind"],
        "mode": mode,
        "retrieved": retrieved,
        "ndcg": {str(k): ndcg_at_k(retrieved, gt, k) for k in K_VALUES},
        "recall": {str(k): recall_at_k(retrieved, gt, k) for k in K_VALUES},
    }


async def run(
    url: str,
    benchmark_path: Path,
    shared_path: Path,
    output_dir: Path,
    parallelism: int,
    smoke: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    bench = json.loads(benchmark_path.read_text())
    shared = set(json.loads(shared_path.read_text())["shared_declarations"])
    rows = _flatten(bench)
    if smoke:
        rows = rows[:24]
    log.info(f"benchmark: {len(rows)} queries; shared decls: {len(shared)}")

    client = StandardClient(url=url)
    sem = asyncio.Semaphore(parallelism)

    per_query: list[dict] = []
    for mode in CONFIGS:
        log.info(f"== running mode={mode} ==")
        tasks = [asyncio.create_task(_run_one(client, r, mode, sem)) for r in rows]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=mode):
            per_query.append(await fut)

    def in_slice(rec: dict, slice_name: str) -> bool:
        return slice_name == "full" or rec["full_name"] in shared

    summary: dict = {}
    for slice_name in ["fair", "full"]:
        summary[slice_name] = {}
        for mode in CONFIGS:
            recs = [r for r in per_query if r["mode"] == mode and in_slice(r, slice_name)]
            summary[slice_name][mode] = summarize(recs, K_VALUES)

    (output_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_query) + "\n"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"wrote {output_dir}/per_query.jsonl  &  summary.json")

    # Print Table-1-style summary on stdout
    print()
    print(f"{'slice':<8} {'config':<16} {'n':>5}  " + "  ".join(f"nDCG@{k:<3}" for k in K_VALUES) + "   " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    for slice_name in ["fair", "full"]:
        for mode in CONFIGS:
            s = summary[slice_name][mode]
            ndcg_str = "  ".join(f"{s[f'ndcg@{k}']:>6.3f}" for k in K_VALUES)
            rec_str = "  ".join(f"{s[f'recall@{k}']:>5.3f}" for k in K_VALUES)
            print(f"{slice_name:<8} {mode:<16} {s['n']:>5}  {ndcg_str}   {rec_str}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Standard-mode endpoint (default: localhost:8000)")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--shared", type=str, default=str(DEFAULT_SHARED))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Run only the first 24 queries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(run(
        url=args.url,
        benchmark_path=Path(args.benchmark),
        shared_path=Path(args.shared),
        output_dir=Path(args.output),
        parallelism=args.parallelism,
        smoke=args.smoke,
    ))


if __name__ == "__main__":
    sys.exit(main())
