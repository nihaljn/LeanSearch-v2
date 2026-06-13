"""Encode the corpus JSONL with the embedder and save a plain vector matrix.

This is the Pascal-friendly replacement for ``build_cuvs``: instead of a cuVS
CAGRA index (which needs compute capability >= 7.0), it writes the normalized
embedding matrix as ``embeddings.npy``. The serving pipeline does an exact
GPU brute-force matmul over it, so no approximate index is needed.

Output layout (consumed by ``pipeline.RetrievalPipeline``):

    <output_dir>/
        metadata.pkl     # {'data': [...], 'embedding_dim', 'num_vectors', 'model_path'}
        texts.pkl        # list[str] — the prompted text encoded for each row
        embeddings.npy   # float16 (N, dim), L2-normalized

Typical use: run this on Colab (fast GPU), then copy the three files into
``data/index/mathlib-v4.28.0-rc1/`` on the serving box. Requires only
``sentence-transformers`` + ``torch`` (no cuVS / cupy / faiss).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from ..config import get, get_path
from .build_cuvs import _encode, _load_jsonl, construct_lean_representation


log = logging.getLogger("leansearchv2.corpus.build_embeddings")


def build(
    jsonl_path: Path,
    output_dir: Path,
    embedder_model_path: str,
    batch_size: int = 32,
    num_gpus: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading corpus from {jsonl_path}")
    data = _load_jsonl(jsonl_path)
    log.info(f"Loaded {len(data)} records")

    texts = [construct_lean_representation(r) for r in data]
    with (output_dir / "texts.pkl").open("wb") as f:
        pickle.dump(texts, f)

    import torch

    n_gpu = num_gpus if num_gpus is not None else torch.cuda.device_count()
    emb = _encode(texts, embedder_model_path, batch_size, n_gpu)  # float32, normalized

    np.save(output_dir / "embeddings.npy", emb.astype(np.float16))

    metadata = {
        "data": data,
        "embedding_dim": int(emb.shape[1]),
        "num_vectors": int(emb.shape[0]),
        "model_path": embedder_model_path,
    }
    with (output_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    log.info(f"OK: wrote {emb.shape[0]} vectors (dim {emb.shape[1]}) to {output_dir}")
    return int(emb.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode corpus JSONL into embeddings.npy")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Corpus JSONL (default: <workdir>/final/mathlib_corpus.jsonl)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: paths.index_db)")
    parser.add_argument("--embedder", type=str, default=None,
                        help="Embedding model path or HF id (default: models.embedder)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Override visible GPU count (default: all visible)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        workdir = Path(get_path("CORPUS_WORKDIR", "paths", "corpus_workdir"))
        jsonl_path = workdir / "final" / "mathlib_corpus.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else Path(get_path("VECTORDB_DIR", "paths", "index_db"))
    embedder = args.embedder or get("EMBEDDING_MODEL_PATH", "models", "embedder")

    n = build(jsonl_path, output_dir, embedder, args.batch_size, args.num_gpus)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
