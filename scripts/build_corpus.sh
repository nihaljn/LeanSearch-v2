#!/usr/bin/env bash
# Build the LeanSearch v2 corpus from scratch.
#
# Pipeline (all four sub-steps; comment / split as needed):
#   1a. jixia_extract     — Mathlib *.lean -> raw .sym/.decl/.mod JSON  (lake env; CPU is enough)
#   1b. merge_to_jsonl    — raw JSON -> merged JSONL                    (CPU-only)
#   1c. informalize       — merged JSONL -> final JSONL (with informal) (CPU + LLM API)
#   2.  build_cuvs        — final JSONL -> cuVS index + metadata.pkl    (requires GPU)
#
# Most users SHOULD NOT need to run this. The default workflow is:
#   - download the pre-built cuVS DB from
#     https://huggingface.co/datasets/FrenzyMath/lsv2-mathlib-v4.28.0-rc1-cuvs
#     into `data/cuvs/mathlib-v4.28.0-rc1/`
#   - run scripts/serve.sh
#
# Settings come from config.yaml (paths.mathlib4 / jixia_bin / corpus_workdir,
# models.embedder, llm.*). Override per invocation via env vars (see each module's main).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Allow running without `pip install -e .`
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PY:-python}"

echo "== Step 1a: jixia extract =="
$PY -m leansearchv2.corpus.jixia_extract

echo "== Step 1b: merge to JSONL =="
$PY -m leansearchv2.corpus.merge_to_jsonl

echo "== Step 1c: informalize (LLM) =="
$PY -m leansearchv2.corpus.informalize

echo "== Step 2: build cuVS index =="
$PY -m leansearchv2.corpus.build_cuvs

echo "Corpus build complete."
