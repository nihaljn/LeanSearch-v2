"""Step 1c: turn the merged jixia JSONL into the final corpus JSONL by
attaching `informal_name` and `informal_description` to every retainable
declaration.

This is the dependency-aware bottom-up informalizer described in §3.1 of
the paper. For each declaration we resolve `typeReferences` and (for
non-Prop declarations) `valueReferences` against the corpus, build a
dependency DAG, topologically sort it, and informalize level-by-level
so that every prompt includes the already-informalized descriptions of
the target's dependencies. Kind-specific Jinja templates (theorem /
definition / instance / technical) ported verbatim from the internal
pipeline drive the prompt construction.

LLM endpoint, model, and key come from `config.yaml::llm` (see
`leansearchv2.llm.LLMClient`).  Tunables (parallelism, context budget,
dependency cap) live under `config.yaml::informalize`.

Compiler-generated identifiers (anonymous numeric components, names
ending in `_proof_*`, `_match_*`, etc.) are filtered out by the
`is_internal` heuristic so they neither consume LLM calls nor pollute
the dependency context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from tqdm.asyncio import tqdm_asyncio

from ..config import get, get_path
from ..llm import LLMClient


log = logging.getLogger("leansearchv2.corpus.informalize")


# Kinds that end up in the published corpus JSONL.
OUTPUT_KINDS = {"theorem", "definition", "instance", "abbrev", "opaque", "axiom"}

# Kinds that get informalized so they can serve as dependency context.
# A theorem's prompt benefits from having `Real`'s informal description
# available even though `Real` itself (kind=inductive) is not written to
# the output corpus.
STRUCTURAL_KINDS = {"constructor", "recursor", "inductive", "classInductive", "structure", "class"}
INFORMALIZE_KINDS = OUTPUT_KINDS | STRUCTURAL_KINDS


SYSTEM_PROMPT = (
    "You are a precise mathlib translator. "
    "Return ONLY a JSON object with keys informal_name and informal_description. "
    "Do not output markdown or extra keys."
)

USER_SUFFIX = (
    "\n\nReturn format (strict JSON only):\n"
    '{"informal_name":"...","informal_description":"..."}'
    "\nIf uncertain, write a conservative but non-empty answer."
)


# ---------------------------------------------------------------------------
# Heuristics


def is_internal(name: list) -> bool:
    """Heuristic for Lean-compiler-generated identifiers.

    Mirrors the project's reference filter: any name with a non-string
    component (anonymous numeric segments) or whose final segment starts
    with `_`, `eq_`, `match_`, or `proof_` is treated as internal and
    dropped from both the retained set and the dependency graph.
    """
    if not name:
        return True
    if any(type(x) is not str for x in name):
        return True
    last = name[-1]
    if any(last.startswith(p) for p in ("_", "eq_", "match_", "proof_")):
        return True
    return False


def _name_key(name: list) -> tuple:
    """Stable hashable key for a LeanName."""
    return tuple(name) if isinstance(name, list) else (name,)


def _pp_name(name: Any) -> str:
    if isinstance(name, list):
        return ".".join(str(x) for x in name)
    return str(name)


# ---------------------------------------------------------------------------
# Token budgeting (copied from the internal pipeline)

_DEFAULT_CHARS_PER_TOKEN = 3.8


def _estimate_tokens(text: str, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def _truncate_by_tokens(
    text: str | None, token_limit: int, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN
) -> str:
    if not text:
        return ""
    if token_limit <= 0:
        return ""
    text = text.strip()
    if _estimate_tokens(text, chars_per_token) <= token_limit:
        return text
    char_limit = int(token_limit * chars_per_token)
    if char_limit <= 3:
        return text[:char_limit]
    return text[: max(0, char_limit - 3)] + "..."


# ---------------------------------------------------------------------------
# JSON-from-LLM extraction


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Template loading


def _template_dir() -> Path:
    """Resolve the packaged template directory. Works for both editable
    installs and regular site-packages installs as long as the templates
    ship as package data (see pyproject.toml::tool.setuptools.package-data)."""
    return Path(__file__).resolve().parent / "templates"


def _build_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.filters["pp_name"] = _pp_name
    return env


def _pick_template(effective_kind: str) -> str:
    if effective_kind == "theorem":
        return "theorem.md.j2"
    if effective_kind in {"definition", "abbrev"}:
        return "definition.md.j2"
    if effective_kind == "instance":
        return "instance.md.j2"
    return "technical_entry.md.j2"


# ---------------------------------------------------------------------------
# Corpus loading + dependency resolution


def _retainable(row: dict) -> bool:
    """A row is retainable iff its kind is in INFORMALIZE_KINDS, it is
    not private, and its name is not an internal/compiler-generated one.

    `row["kind"]` is already the effective kind: merge_to_jsonl sets it
    from decl.kind when present and falls back to sym.kind otherwise, so
    `example` entries (decl-level kind, no informal-corpus value) are
    filtered out here even though their sym-level kind is `theorem` or
    `definition`."""
    kind = (row.get("kind") or "").strip()
    if kind not in INFORMALIZE_KINDS:
        return False
    if row.get("visibility") == "private":
        return False
    if is_internal(row.get("name") or []):
        return False
    return True


def _load_corpus(input_jsonl: Path) -> tuple[list[dict], dict[tuple, int]]:
    """Load the merged JSONL, filter to retainable rows, build a
    name -> index lookup. Returns (rows, name_index)."""
    rows: list[dict] = []
    index: dict[tuple, int] = {}
    with input_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _retainable(row):
                continue
            key = _name_key(row.get("name") or [])
            if not key or key in index:
                # Skip duplicate names; first wins. Mathlib has a small
                # number of legitimate duplicates from re-exports.
                continue
            index[key] = len(rows)
            rows.append(row)
    return rows, index


def _effective_kind(row: dict) -> str:
    return (row.get("kind") or "").strip() or "unknown"


def _is_prop(row: dict) -> bool:
    """A row is treated as a Prop iff jixia says so. Fallback: theorem-kind."""
    if row.get("isProp") is True:
        return True
    return _effective_kind(row) == "theorem"


def _resolve_deps(
    rows: list[dict], index: dict[tuple, int]
) -> list[list[int]]:
    """For each row, return the list of (in-index) row positions it
    depends on. type-refs always count; value-refs only count for
    non-Prop declarations (mirrors run_informal_generation.fetch_dependencies).
    Internal refs and self-refs are dropped."""
    deps: list[list[int]] = [[] for _ in rows]
    for src_idx, row in enumerate(rows):
        seen: set[int] = set()
        type_refs = row.get("typeReferences") or []
        value_refs = row.get("valueReferences") or []
        use_value = not _is_prop(row)
        for ref in type_refs:
            key = _name_key(ref)
            if is_internal(list(key)):
                continue
            tgt = index.get(key)
            if tgt is None or tgt == src_idx or tgt in seen:
                continue
            seen.add(tgt)
        if use_value:
            for ref in value_refs:
                key = _name_key(ref)
                if is_internal(list(key)):
                    continue
                tgt = index.get(key)
                if tgt is None or tgt == src_idx or tgt in seen:
                    continue
                seen.add(tgt)
        deps[src_idx] = sorted(seen)
    return deps


def _compute_levels(deps: list[list[int]]) -> list[int]:
    """Kahn's algorithm. Cycle-bound nodes (mutual recursion in Lean)
    are placed at one level above the highest non-cyclic level so the
    informalizer still emits them, even if their in-cycle peers are
    invisible to each other."""
    n = len(deps)
    indeg = [0] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for src, ts in enumerate(deps):
        for t in ts:
            indeg[src] += 1
            children[t].append(src)
    level = [-1] * n
    q: deque[int] = deque()
    for i in range(n):
        if indeg[i] == 0:
            level[i] = 0
            q.append(i)
    while q:
        u = q.popleft()
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                # node's actual level is 1 + max(level of any dep)
                lvl = 0
                for d in deps[v]:
                    if level[d] >= 0:
                        lvl = max(lvl, level[d] + 1)
                level[v] = lvl
                q.append(v)
    # Cycle nodes: stick them at one level above the highest finite level.
    cycle_level = (max((lv for lv in level if lv >= 0), default=-1)) + 1
    for i in range(n):
        if level[i] < 0:
            level[i] = cycle_level
    return level


# ---------------------------------------------------------------------------
# Prompt construction + budget shrinking


def _build_input_obj(
    row: dict,
    level: int,
    deps_idx: list[int],
    rows: list[dict],
    informal: dict[tuple, dict[str, str]],
    cfg: dict[str, int],
) -> tuple[str, str, dict[str, Any]]:
    """Build a (template_name, effective_kind, input_obj) tuple for the
    Jinja env. Dep entries that have not yet been informalized (failed
    in an earlier level or filtered out) are silently dropped."""
    eff_kind = _effective_kind(row)
    template_name = _pick_template(eff_kind)

    signature_raw = row.get("signature") or row.get("type") or ""
    value_raw = row.get("value") or ""
    docstring_raw = row.get("docstring") or ""

    signature = _truncate_by_tokens(signature_raw, cfg["max_signature_tokens"])
    value = _truncate_by_tokens(value_raw, cfg["max_value_tokens"])
    docstring = _truncate_by_tokens(docstring_raw, cfg["max_docstring_tokens"])

    dep_items: list[dict[str, Any]] = []
    for di in deps_idx:
        dep_row = rows[di]
        key = _name_key(dep_row.get("name") or [])
        info = informal.get(key)
        if not info:
            continue
        desc = _truncate_by_tokens(info.get("informal_description") or "", cfg["max_dependency_item_tokens"])
        if not desc:
            continue
        dep_items.append(
            {
                "name": dep_row.get("name") or [],
                "description": desc,
                "informal_name": info.get("informal_name") or "",
                "informal_description": desc,
            }
        )
        if len(dep_items) >= cfg["max_dependency_items"]:
            break

    module_str = row.get("module") or ""
    is_prop = _is_prop(row)
    input_obj = {
        "header": f"Module: {module_str}\nEntry level: {level}\nKind: {eff_kind}",
        "kind": eff_kind,
        "docstring": docstring,
        "dependency": dep_items,
        "neighbor": [],
        "name": row.get("name") or [],
        "signature": signature,
        "value_matters": (not is_prop) and bool(value.strip()),
        "value": value,
    }
    return template_name, eff_kind, input_obj


def _render(env: Environment, template_name: str, input_obj: dict[str, Any]) -> str:
    return env.get_template(template_name).render(input=input_obj)


def _shrink_to_budget(
    env: Environment,
    template_name: str,
    input_obj: dict[str, Any],
    max_input_tokens: int,
    chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
) -> tuple[str, dict[str, Any]]:
    """Render the prompt; if it exceeds max_input_tokens, progressively
    drop dependencies, then halve docstring/value/signature. Ported from
    new_jixia.scripts.rag_informal.run_informal_generation."""
    prompt = _render(env, template_name, input_obj)
    est = _estimate_tokens(prompt, chars_per_token)
    if est <= max_input_tokens:
        return prompt, input_obj

    deps = list(input_obj.get("dependency", []))
    while est > max_input_tokens and deps:
        shrink = max(1, len(deps) // 8)
        deps = deps[:-shrink]
        input_obj["dependency"] = deps
        prompt = _render(env, template_name, input_obj)
        est = _estimate_tokens(prompt, chars_per_token)

    if est > max_input_tokens:
        for field in ("docstring", "value", "signature"):
            v = input_obj.get(field) or ""
            if not v:
                continue
            cur = _estimate_tokens(v, chars_per_token)
            input_obj[field] = _truncate_by_tokens(v, max(64, cur // 2), chars_per_token)
            prompt = _render(env, template_name, input_obj)
            est = _estimate_tokens(prompt, chars_per_token)
            if est <= max_input_tokens:
                break
    return prompt, input_obj


# ---------------------------------------------------------------------------
# Per-row processing


async def _informalize_one(
    llm: LLMClient,
    env: Environment,
    row: dict,
    level: int,
    deps_idx: list[int],
    rows: list[dict],
    informal: dict[tuple, dict[str, str]],
    cfg: dict[str, int],
    sem: asyncio.Semaphore,
) -> dict[str, str] | None:
    template_name, _kind, input_obj = _build_input_obj(row, level, deps_idx, rows, informal, cfg)
    prompt_text, _ = _shrink_to_budget(env, template_name, input_obj, cfg["max_input_tokens"])
    user_text = prompt_text + USER_SUFFIX
    async with sem:
        try:
            content = await llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=cfg["max_output_tokens"],
            )
        except Exception as e:
            log.warning(f"LLM call failed for {_pp_name(row.get('name'))}: {e}")
            return None
    obj = _extract_json(content)
    if not isinstance(obj, dict):
        log.warning(f"Bad LLM output for {_pp_name(row.get('name'))}: {content[:200]!r}")
        return None
    name = str(obj.get("informal_name") or "").strip()
    desc = str(obj.get("informal_description") or "").strip()
    if not name or not desc:
        log.warning(f"Empty informal_name/desc for {_pp_name(row.get('name'))}")
        return None
    return {"informal_name": name, "informal_description": desc}


def _to_final_row(row: dict, info: dict[str, str], index: int) -> dict:
    module_str = row.get("module") or ""
    module_name = module_str.split(".") if module_str else []
    return {
        "name": row.get("name") or [],
        "module_name": module_name,
        "kind": row.get("kind") or "",
        "type": row.get("type") or "",
        "index": index,
        "signature": row.get("signature") or "",
        "value": row.get("value") or "",
        "informal_name": info["informal_name"],
        "informal_description": info["informal_description"],
    }


# ---------------------------------------------------------------------------
# Top-level driver


def _load_cfg() -> dict[str, int]:
    """Read informalize tunables from config.yaml::informalize with
    paper-matching defaults."""
    def _cfg(name: str, default: int) -> int:
        v = get(f"INFORMALIZE_{name.upper()}", "informalize", name, default=default)
        return int(v) if v is not None else default
    return {
        "parallelism": _cfg("parallelism", 16),
        "max_input_tokens": _cfg("max_input_tokens", 200_000),
        "max_output_tokens": _cfg("max_output_tokens", 16_000),
        "max_signature_tokens": _cfg("max_signature_tokens", 3500),
        "max_value_tokens": _cfg("max_value_tokens", 6000),
        "max_docstring_tokens": _cfg("max_docstring_tokens", 800),
        "max_dependency_items": _cfg("max_dependency_items", 96),
        "max_dependency_item_tokens": _cfg("max_dependency_item_tokens", 400),
    }


async def informalize(
    input_jsonl: Path, output_jsonl: Path, limit: int | None = None,
) -> int:
    cfg = _load_cfg()
    profile = str(get("INFORMALIZE_LLM", "informalize", "llm", default="openai"))
    llm = LLMClient(profile)
    env = _build_jinja_env()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading merged corpus from {input_jsonl}")
    rows, index = _load_corpus(input_jsonl)
    if limit:
        rows = rows[:limit]
        index = {k: i for k, i in index.items() if i < limit}
    log.info(f"Loaded {len(rows)} retainable rows (after kind+visibility+is_internal filter)")

    log.info("Resolving dependency edges (type-refs always, value-refs for non-Prop)")
    deps = _resolve_deps(rows, index)
    log.info("Computing topological levels")
    levels = _compute_levels(deps)

    max_level = max(levels) if levels else 0
    log.info(f"Levels span 0..{max_level}; processing bottom-up")

    by_level: dict[int, list[int]] = defaultdict(list)
    for i, lv in enumerate(levels):
        by_level[lv].append(i)

    informal: dict[tuple, dict[str, str]] = {}
    sem = asyncio.Semaphore(cfg["parallelism"])

    n_attempted = 0
    n_succeeded = 0
    for lv in sorted(by_level.keys()):
        batch = by_level[lv]
        n_attempted += len(batch)
        tasks = [
            asyncio.create_task(
                _informalize_one(llm, env, rows[i], lv, deps[i], rows, informal, cfg, sem)
            )
            for i in batch
        ]
        results = await tqdm_asyncio.gather(*tasks, desc=f"level {lv} ({len(batch)})", leave=False)
        for i, res in zip(batch, results):
            if res is None:
                continue
            key = _name_key(rows[i].get("name") or [])
            informal[key] = res
            n_succeeded += 1
        log.info(f"level {lv}: {sum(1 for r in results if r is not None)}/{len(batch)} succeeded")

    # Write OUTPUT_KINDS subset only, preserving original row order.
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if (row.get("kind") or "") not in OUTPUT_KINDS:
                continue
            key = _name_key(row.get("name") or [])
            info = informal.get(key)
            if info is None:
                continue
            out.write(json.dumps(_to_final_row(row, info, written), ensure_ascii=False) + "\n")
            written += 1
    log.info(
        f"Informalized {n_succeeded}/{n_attempted} declarations; "
        f"wrote {written} rows ({sorted(OUTPUT_KINDS)} only) -> {output_jsonl}"
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Dependency-aware informalize for the Mathlib corpus")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Corpus workdir (default: paths.corpus_workdir)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input merged JSONL (default: <workdir>/merged/mathlib_declarations.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output final JSONL (default: <workdir>/final/mathlib_corpus.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N retained rows (for smoke tests).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    workdir = Path(args.workdir) if args.workdir else Path(get_path("CORPUS_WORKDIR", "paths", "corpus_workdir"))
    inp = Path(args.input) if args.input else workdir / "merged" / "mathlib_declarations.jsonl"
    out = Path(args.output) if args.output else workdir / "final" / "mathlib_corpus.jsonl"
    n = asyncio.run(informalize(inp, out, limit=args.limit))
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
