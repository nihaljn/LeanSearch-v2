# Building the corpus embeddings on Colab (4B, Pascal-friendly serving)

The serving box (3× GTX 1080 Ti, Pascal sm_61) can't run cuVS/FAISS-GPU, so
we use an exact GPU brute-force matmul over a plain embedding matrix. Encoding
the full Mathlib corpus with Qwen3-Embedding-4B is slow on Pascal, so do it on
a Colab GPU and copy three files back.

## On Colab

```python
!pip install -q sentence-transformers huggingface_hub

from huggingface_hub import snapshot_download
# JSONL corpus (text + metadata, one record per declaration)
snapshot_download("FrenzyMath/lsv2-mathlib-v4.28.0-rc1-jsonl",
                  repo_type="dataset", local_dir="corpus")
# -> find the .jsonl inside corpus/ (e.g. corpus/mathlib_corpus.jsonl)

# Clone this repo to reuse the exact encode + prompt logic:
!git clone https://github.com/<you>/LeanSearch-v2 && cd LeanSearch-v2 && pip install -e . --no-deps

import sys; sys.path.insert(0, "LeanSearch-v2/src")
from pathlib import Path
from leansearchv2.corpus.build_embeddings import build

build(
    jsonl_path=Path("corpus/mathlib_corpus.jsonl"),   # adjust to actual filename
    output_dir=Path("out"),
    embedder_model_path="Qwen/Qwen3-Embedding-4B",
    batch_size=64,        # bump if the Colab GPU has headroom
    num_gpus=1,
)
```

This writes `out/embeddings.npy` (float16, L2-normalized), `out/metadata.pkl`,
and `out/texts.pkl`. Download all three (e.g. zip `out/` and pull via the
Files pane or `google.colab.files.download`).

## On the serving box

Copy the three files into the index dir configured in `config.yaml`
(`paths.index_db`, default below):

```
data/index/mathlib-v4.28.0-rc1/
    embeddings.npy
    metadata.pkl
    texts.pkl
```

Then serve (use the project venv):

```bash
source .venv/bin/activate
./scripts/serve.sh            # uvicorn on 0.0.0.0:8000
```

Smoke-test:

```bash
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": ["the order of a group element divides the order of the group"], "num_results": 10}'
```

## Notes

- The embedder used here (4B) **must** match the one the server loads
  (`models.embedder` in `config.yaml`) — query and corpus vectors have to come
  from the same model. Both are already set to `Qwen/Qwen3-Embedding-4B`.
- `embeddings.npy` row order must line up with `metadata.pkl["data"]`; the
  server asserts the counts match on startup.
