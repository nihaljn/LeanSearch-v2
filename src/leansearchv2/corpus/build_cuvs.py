"""Step 2: encode every record in the corpus JSONL with the embedder and
build a cuVS CAGRA index under `<cuvs_db>/`.

Output layout (matches `pipeline.RetrievalPipeline`'s loader):

    <cuvs_db>/
        metadata.pkl    # {'data': [...], 'embedding_dim': int, 'num_vectors': int, 'model_path': str}
        texts.pkl       # list[str] — the prompted text encoded for each row
        cuvs_index.bin  # cuVS CAGRA binary index

Requires a GPU. Heavy imports (torch / cuvs / cupy / sentence_transformers)
are deferred so this module can be imported without those packages present.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..config import get, get_path


log = logging.getLogger("leansearchv2.corpus.build_cuvs")


def construct_lean_representation(record: dict[str, Any]) -> str:
    """Canonical text representation used for embedding.

    Preserves the exact format that produced the published cuVS corpus:
    kind-aware prefix, informal-content block, formal-content block. The
    `kind_part` interpolation matches the original (the list-repr quirk is
    intentional — it is what the deployed embedder learned to expect).
    """
    kind_part: list[str] = []
    if record.get("kind"):
        kind_part.append(record["kind"])

    informal_parts: list[str] = []
    if record.get("informal_name"):
        informal_parts.append(record["informal_name"])
    if record.get("informal_description"):
        desc = record.get("informal_description", "") or ""
        if len(desc) > 10000:
            desc = desc[:10000] + "...(truncated)"
        informal_parts.append(desc)
    informal_text_p = ": ".join(informal_parts)
    informal_text = f"{kind_part}: {informal_text_p}" if kind_part else informal_text_p

    lean_parts: list[str] = []
    if record.get("name"):
        lean_parts.append(".".join(record["name"]))
    if record.get("type"):
        lean_parts.append(record["type"])
    kind = record.get("kind") or ""
    if record.get("value") and kind in ("definition", "instance"):
        lean_parts.append(record["value"])
        lean_text = " ".join(lean_parts)
    else:
        lean_text = " ".join(lean_parts) + " := by sorry"

    if kind == "theorem":
        header = "Represent the following mathematical theorem in lean repository for semantic search"
    elif kind == "definition":
        header = "Represent the following mathematical definition in lean repository for semantic search"
    elif kind == "instance":
        header = "Represent the following typeclass instance in lean repository for semantic search"
    else:
        header = "Represent the following lean content for semantic search"

    return f"{header}: \n Informal content: \n {informal_text} \n Formal content: \n {lean_text}"


def _encode_on_gpu_worker(args):
    gpu_id, texts, model_path, batch_size = args
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=f"cuda:{gpu_id}")
    out: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = model.encode(texts[i:i + batch_size], convert_to_numpy=True, show_progress_bar=False)
        out.append(batch)
        cur = i // batch_size + 1
        if cur % max(1, total_batches // 10) == 0 or cur == total_batches:
            print(f"[GPU {gpu_id}] {cur}/{total_batches} batches", flush=True)
    return gpu_id, np.vstack(out)


def _load_jsonl(path: Path) -> list[dict]:
    data: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load jsonl", unit="row"):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def _encode(texts: list[str], model_path: str, batch_size: int, num_gpus: int):
    import numpy as np
    import torch

    if num_gpus <= 0 or not torch.cuda.is_available():
        raise RuntimeError("build_cuvs requires at least one CUDA device")
    actual = min(num_gpus, torch.cuda.device_count())
    log.info(f"Encoding {len(texts)} texts on {actual} GPU(s)")
    if actual == 1:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, device="cuda:0")
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    else:
        chunk = (len(texts) + actual - 1) // actual
        tasks = []
        for g in range(actual):
            start = g * chunk
            end = min(len(texts), start + chunk)
            if start < end:
                tasks.append((g, texts[start:end], model_path, batch_size))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            results = pool.map(_encode_on_gpu_worker, tasks)
        results.sort(key=lambda x: x[0])
        emb = np.vstack([r[1] for r in results])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")


def _build_cagra_index(emb, out_path: Path, metric: str = "cosine"):
    import cupy as cp
    from cuvs.neighbors import cagra

    log.info(f"Building cuVS CAGRA index ({emb.shape}, metric={metric})")
    t0 = time.time()
    vectors_gpu = cp.asarray(emb)
    params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=128,
        graph_degree=64,
        build_algo="ivf_pq",
    )
    index = cagra.build(params, vectors_gpu)
    log.info(f"  built in {time.time() - t0:.1f}s; saving to {out_path}")
    cagra.save(str(out_path), index)


def build(
    jsonl_path: Path,
    output_dir: Path,
    embedder_model_path: str,
    batch_size: int = 32,
    num_gpus: int | None = None,
    metric: str = "cosine",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading corpus from {jsonl_path}")
    data = _load_jsonl(jsonl_path)
    log.info(f"Loaded {len(data)} records")

    texts = [construct_lean_representation(r) for r in tqdm(data, desc="prompt", unit="row")]
    with (output_dir / "texts.pkl").open("wb") as f:
        pickle.dump(texts, f)

    import torch
    n_gpu = num_gpus if num_gpus is not None else torch.cuda.device_count()
    emb = _encode(texts, embedder_model_path, batch_size, n_gpu)

    metadata = {
        "data": data,
        "embedding_dim": int(emb.shape[1]),
        "num_vectors": int(emb.shape[0]),
        "model_path": embedder_model_path,
    }
    with (output_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    _build_cagra_index(emb, output_dir / "cuvs_index.bin", metric=metric)
    log.info(f"OK: wrote {emb.shape[0]} vectors (dim {emb.shape[1]}) to {output_dir}")
    return int(emb.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode corpus JSONL and build cuVS index")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Final corpus JSONL (default: <workdir>/final/mathlib_corpus.jsonl)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output cuVS DB dir (default: paths.cuvs_db)")
    parser.add_argument("--embedder", type=str, default=None,
                        help="Embedding model path or HF id (default: models.embedder)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Override visible GPU count (default: serve.num_gpus or all visible)")
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "inner_product", "sqeuclidean"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        workdir = Path(get_path("CORPUS_WORKDIR", "paths", "corpus_workdir"))
        jsonl_path = workdir / "final" / "mathlib_corpus.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else Path(get_path("VECTORDB_DIR", "paths", "cuvs_db"))
    embedder = args.embedder or get("EMBEDDING_MODEL_PATH", "models", "embedder")
    num_gpus = args.num_gpus
    if num_gpus is None:
        cfg_ng = get("NUM_GPUS", "serve", "num_gpus")
        num_gpus = int(cfg_ng) if cfg_ng is not None else None

    n = build(jsonl_path, output_dir, embedder, args.batch_size, num_gpus, args.metric)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
