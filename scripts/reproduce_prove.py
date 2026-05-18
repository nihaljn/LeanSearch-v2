"""Reproduce the LeanSearch v2 rows of Table 3 (Prove task on FATE-H and
MathlibMPR-Prop).

Runs the simple reflection loop against a running standard-mode retriever
endpoint and an in-process Lean 4 REPL (lean-interact). Two retriever
configurations are evaluated by default, matching the LSv2 rows of Table 3:

  - standard  : per-reflection-round query → /search; matches the paper's
                LSv2 (standard mode) row.
  - reasoning : one-shot reasoning prefetch on the theorem statement;
                matches the paper's LSv2 (reasoning mode) row.

Pass `--retriever-modes none` to additionally produce the no-retrieval
baseline row (paper's "no retrieval" row).

Datasets (CLI):
  --benchmark fate_h            -> benchmark/FATE-H.jsonl (100 problems)
  --benchmark mathlibmpr_prop   -> benchmark/MathlibMPR.json filtered to
                                   MathlibMPR_Prop_ids.txt (50 problems)
  --benchmark all               -> both, scored separately

Outputs (default `reports/prove/<benchmark>_<mode>/`):
  - per_problem.jsonl   one record per problem with full attempt history
  - summary.json        {n, solved, success_rate}
  - prints the Table-3-style summary on stdout

Requires:
  - LLM API key (env var named by `llm.api_key_env` in config.yaml)
  - lean-interact + a Lean 4 toolchain reachable on PATH; install with
    `pip install -e '.[lean]'`. Pass `--dry-run` to skip Lean verification
    (useful for exercising the LLM / retriever path without a Lean install).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.asyncio import tqdm_asyncio

from leansearchv2 import StandardClient
from leansearchv2.config import get
from leansearchv2.prove import LeanInteractVerifier, ProveLLMs, ProveProblem, run_prove


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FATE_H = REPO_ROOT / "benchmark" / "FATE-H.jsonl"
DEFAULT_MPR = REPO_ROOT / "benchmark" / "MathlibMPR.json"
DEFAULT_MPR_IDS = REPO_ROOT / "benchmark" / "MathlibMPR_Prop_ids.txt"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "prove"


log = logging.getLogger("reproduce_prove")


def _load_fate_h(path: Path) -> list[ProveProblem]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out: list[ProveProblem] = []
    for r in rows:
        header = r.get("header", "")
        out.append(ProveProblem(
            problem_id=r.get("name") or r.get("problem_id") or f"row_{len(out)}",
            formal_statement=(header + r["formal_statement"]).strip(),
            informal_statement=r.get("informal_statement", ""),
            header=header,
        ))
    return out


def _load_mathlibmpr_prop(json_path: Path, ids_path: Path) -> list[ProveProblem]:
    by_id = {r["id"]: r for r in json.loads(json_path.read_text())}
    ids: list[str] = []
    for line in ids_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    out: list[ProveProblem] = []
    for mid in ids:
        r = by_id.get(mid)
        if r is None:
            log.warning(f"missing id {mid} in MathlibMPR.json")
            continue
        out.append(ProveProblem(
            problem_id=mid,
            formal_statement=r["formal_statement"],
            informal_statement=r.get("NL_main_result", ""),
            header="",
        ))
    return out


class _DryRunVerifier:
    """No-op verifier; the first attempt is always reported as failed so
    each problem produces a single trial record without invoking Lean."""

    async def verify(self, code: str, timeout_s: int = 600):
        from leansearchv2.prove.verifier import VerifyResult
        return VerifyResult(complete=False, error_msg="(dry-run: verifier disabled)", has_sorry=False)

    async def close(self) -> None:
        pass


async def _run_problem(
    problem: ProveProblem,
    *,
    llms: ProveLLMs,
    retriever: StandardClient | None,
    verifier,
    retriever_mode: str,
    reflection_rounds: int,
    sem: asyncio.Semaphore,
    verify_timeout_s: int,
) -> dict[str, Any]:
    async with sem:
        try:
            result = await run_prove(
                problem,
                llms,
                retriever,
                verifier,
                retriever_mode=retriever_mode,
                reflection_rounds=reflection_rounds,
                verify_timeout_s=verify_timeout_s,
            )
            return result.to_dict()
        except Exception as e:
            log.warning(f"FAIL {problem.problem_id}: {type(e).__name__}: {e}")
            return {
                "id": problem.problem_id,
                "success": False,
                "rounds_used": 0,
                "retriever_mode": retriever_mode,
                "final_error": f"{type(e).__name__}: {e}",
                "attempts": [],
            }


async def run_one_cell(
    *,
    benchmark_name: str,
    problems: list[ProveProblem],
    retriever_mode: str,
    llms: ProveLLMs,
    retriever: StandardClient | None,
    verifier,
    reflection_rounds: int,
    parallelism: int,
    verify_timeout_s: int,
    output_dir: Path,
) -> dict[str, Any]:
    cell_dir = output_dir / f"{benchmark_name}_{retriever_mode}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"== {benchmark_name} / retriever_mode={retriever_mode} : {len(problems)} problems ==")
    sem = asyncio.Semaphore(parallelism)
    tasks = [
        asyncio.create_task(_run_problem(
            p, llms=llms, retriever=retriever, verifier=verifier,
            retriever_mode=retriever_mode, reflection_rounds=reflection_rounds,
            sem=sem, verify_timeout_s=verify_timeout_s,
        ))
        for p in problems
    ]
    results: list[dict[str, Any]] = []
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"{benchmark_name}/{retriever_mode}"):
        results.append(await fut)

    n_solved = sum(1 for r in results if r["success"])
    summary = {
        "benchmark": benchmark_name,
        "retriever_mode": retriever_mode,
        "n": len(results),
        "solved": n_solved,
        "success_rate": n_solved / max(len(results), 1),
    }
    (cell_dir / "per_problem.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
    )
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = args.url or get("RETRIEVER_URL", "retriever", "url", default="http://localhost:8000")
    reflection_rounds = args.reflection_rounds if args.reflection_rounds is not None else int(
        get("PROVE_REFLECTION_ROUNDS", "prove", "reflection_rounds", default=8)
    )

    modes = args.retriever_modes.split(",")
    llms = ProveLLMs.from_config(with_reasoning="reasoning" in modes)
    retriever = StandardClient(url=url)
    if args.dry_run:
        verifier: Any = _DryRunVerifier()
    else:
        verifier = LeanInteractVerifier(
            project_dir=args.lean_project_dir,
            lean_version=args.lean_version,
        )

    benchmarks: list[tuple[str, list[ProveProblem]]] = []
    if args.benchmark in ("fate_h", "all"):
        problems = _load_fate_h(Path(args.fate_h_path))
        if args.smoke:
            problems = problems[:3]
        benchmarks.append(("fate_h", problems))
    if args.benchmark in ("mathlibmpr_prop", "all"):
        problems = _load_mathlibmpr_prop(Path(args.mpr_path), Path(args.mpr_ids_path))
        if args.smoke:
            problems = problems[:3]
        benchmarks.append(("mathlibmpr_prop", problems))

    summaries: list[dict] = []
    try:
        for bname, problems in benchmarks:
            for mode in modes:
                summaries.append(await run_one_cell(
                    benchmark_name=bname,
                    problems=problems,
                    retriever_mode=mode,
                    llms=llms,
                    retriever=retriever if mode != "none" else None,
                    verifier=verifier,
                    reflection_rounds=reflection_rounds,
                    parallelism=args.parallelism,
                    verify_timeout_s=args.verify_timeout_s,
                    output_dir=output_dir,
                ))
    finally:
        try:
            await verifier.close()
        except Exception:
            pass

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print()
    print(f"{'benchmark':<18} {'retriever_mode':<14} {'n':>4}  {'#solved':>8}  {'%':>6}")
    for s in summaries:
        print(f"{s['benchmark']:<18} {s['retriever_mode']:<14} {s['n']:>4}  {s['solved']:>8}  {s['success_rate']*100:>5.1f}%")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None,
                        help="Standard-mode endpoint (default: retriever.url from config)")
    parser.add_argument("--benchmark", choices=["fate_h", "mathlibmpr_prop", "all"], default="all")
    parser.add_argument("--retriever-modes", default="standard,reasoning",
                        help="Comma-separated subset of {none,standard,reasoning}")
    parser.add_argument("--fate-h-path", default=str(DEFAULT_FATE_H))
    parser.add_argument("--mpr-path", default=str(DEFAULT_MPR))
    parser.add_argument("--mpr-ids-path", default=str(DEFAULT_MPR_IDS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--reflection-rounds", type=int, default=None,
                        help="Override prove.reflection_rounds from config (default 8 per paper)")
    parser.add_argument("--verify-timeout-s", type=int, default=600)
    parser.add_argument("--lean-project-dir", default=None,
                        help="Path to a Lake project for lean-interact (default: temp project with Mathlib)")
    parser.add_argument("--lean-version", default=None,
                        help="Lean version when lean-project-dir is not given (default: v4.28.0-rc1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Lean verification (still calls LLM + retriever)")
    parser.add_argument("--smoke", action="store_true",
                        help="Process only the first 3 problems per benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
