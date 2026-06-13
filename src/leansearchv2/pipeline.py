"""LeanSearch v2 standard-mode retrieval pipeline.

Qwen3-Embedding-4B for query encoding, an exact GPU brute-force matmul for
vector search (the corpus fits comfortably in VRAM, so no approximate index
is needed), and Qwen3-Reranker-4B (HuggingFace, one replica per GPU) for
cross-encoder rerank.

The original release used cuVS CAGRA, but cuVS/RAPIDS requires compute
capability >= 7.0 and so cannot run on Pascal (e.g. GTX 1080 Ti). A
normalized inner-product matmul + top-k over the corpus matrix is exact and
runs in ~20 ms on these cards.
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .standard_client import ResultData, SearchResult


_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RetrievalPipeline:
    INSTRUCTION = (
        "Retrieve the informal + formal representation of items in Mathlib4 "
        "that is mathematically relevant to the query. If the query ask for "
        "theorem or lemma, you shall try to find the entry that starts with "
        "theorem. If the query ask for a definition, you shall try to find "
        "the entry that starts with either definition or instance."
    )

    def __init__(
        self,
        vectordb_dir: str,
        embedding_model_path: str,
        reranker_model_path: str,
        num_gpus: int | None = None,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.vectordb_dir = Path(vectordb_dir)
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.gpu_memory_utilization = gpu_memory_utilization

        print(f"Loading vector database from {self.vectordb_dir}")

        with open(self.vectordb_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.data = metadata["data"]
        self.embedding_dim = metadata["embedding_dim"]

        with open(self.vectordb_dir / "texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.num_gpus = num_gpus if num_gpus is not None else total_gpus
        self.num_gpus = max(0, min(self.num_gpus, total_gpus))

        if self.num_gpus == 0:
            raise RuntimeError("No GPU available. This pipeline requires GPU.")

        self.embedding_device = "cuda:0"
        # Corpus matrix lives on the embedding device so the query/corpus
        # matmul is local (no cross-device copy). ~1.4 GiB for the full
        # Mathlib corpus in fp16, on top of the embedder's ~7.7 GiB.
        self.search_device = self.embedding_device
        self._load_embeddings()
        self.reranker_devices = [f"cuda:{i}" for i in range(1, self.num_gpus)]

        print(f"\n{'=' * 60}")
        print(f"GPU Allocation:")
        print(f"  Available: {total_gpus}, Using: {self.num_gpus}")
        print(f"  Embedding: {self.embedding_device}")
        print(f"  Vector index: GPU brute-force matmul (exact, on {self.search_device})")
        print(f"  Reranker: HuggingFace Qwen3-Reranker (yes/no 2-way log_softmax)")
        print(f"  Reranker devices: {self.reranker_devices}")
        print(f"{'=' * 60}\n")

        self._load_models()

    def _load_embeddings(self) -> None:
        """Load the normalized corpus embedding matrix onto ``search_device``.

        Expects ``<vectordb_dir>/embeddings.npy`` of shape (N, dim) produced
        by ``leansearchv2.corpus.build_embeddings`` with the same embedder as
        ``models.embedder``. Vectors are assumed already L2-normalized; we
        re-normalize defensively and keep them in fp16 to halve VRAM.
        """
        emb_path = self.vectordb_dir / "embeddings.npy"
        if not emb_path.exists():
            raise RuntimeError(
                f"Corpus embeddings not found at {emb_path}. "
                f"Build them with `python -m leansearchv2.corpus.build_embeddings`."
            )

        print(f"Loading corpus embeddings from {emb_path}...")
        emb = np.load(emb_path)
        if emb.shape[0] != len(self.data):
            raise RuntimeError(
                f"embeddings.npy has {emb.shape[0]} rows but metadata has "
                f"{len(self.data)} records; rebuild the index."
            )
        t = torch.from_numpy(np.ascontiguousarray(emb)).to(
            self.search_device, dtype=torch.float16
        )
        t = torch.nn.functional.normalize(t, dim=1)
        self.corpus_emb = t  # (N, dim) fp16 on search_device
        gib = t.element_size() * t.nelement() / (1024**3)
        print(f"Corpus matrix: {tuple(t.shape)} fp16 on {self.search_device} ({gib:.2f} GiB)")

    def _get_gpu_free_memory(self, gpu_idx: int) -> tuple[float, float]:
        free, total = torch.cuda.mem_get_info(gpu_idx)
        return free / (1024**3), total / (1024**3)

    def _load_models(self) -> None:
        import sys

        torch.cuda.empty_cache()
        free_gib, total_gib = self._get_gpu_free_memory(0)
        print(f"GPU memory before embedding: {free_gib:.1f}/{total_gib:.1f} GiB free")
        sys.stdout.flush()

        print(f"Loading embedding model from {self.embedding_model_path}...")
        sys.stdout.flush()
        self.embedding_model = SentenceTransformer(
            self.embedding_model_path, device=self.embedding_device
        )
        print("Embedding model loaded")
        sys.stdout.flush()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(0.5)

        print(f"Loading reranker tokenizer from {self.reranker_model_path}...")
        sys.stdout.flush()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_path, padding_side="left"
        )
        self.yes_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.reranker_tokenizer.encode(
            _PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.reranker_tokenizer.encode(
            _SUFFIX, add_special_tokens=False
        )
        self.max_model_len = 8192
        self.body_max_len = (
            self.max_model_len - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        print(f"Loading reranker replicas on {self.reranker_devices}...")
        sys.stdout.flush()
        self.reranker_models: list = []
        for d in self.reranker_devices:
            print(f"  loading reranker on {d}")
            sys.stdout.flush()
            m = (
                AutoModelForCausalLM.from_pretrained(
                    self.reranker_model_path,
                    torch_dtype=torch.float16 if d.startswith("cuda") else torch.float32,
                )
                .to(d)
                .eval()
            )
            self.reranker_models.append(m)

        print(f"\n{'=' * 60}")
        print(f"All {len(self.reranker_models)} reranker replica(s) ready")
        print(f"{'=' * 60}")

    def _search_vectors(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k inner-product search via a brute-force GPU matmul.

        ``query_embedding`` is (1, dim) and L2-normalized; the corpus matrix
        is normalized too, so inner product == cosine similarity. Returns
        (scores, indices), both (1, k), ordered by descending similarity.
        """
        q = torch.from_numpy(np.ascontiguousarray(query_embedding)).to(
            self.search_device, dtype=self.corpus_emb.dtype
        )
        with torch.no_grad():
            scores = q @ self.corpus_emb.T  # (1, N)
            k = min(k, self.corpus_emb.shape[0])
            vals, idx = torch.topk(scores, k, dim=1)
        return vals.float().cpu().numpy(), idx.cpu().numpy()

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: list[str], device: str):
        """Tokenize pairs, prepend prefix + append suffix tokens, pad to a
        batch tensor on ``device``. Documents are truncated from the back so
        prompts fit within ``max_model_len``.
        """
        inputs = self.reranker_tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            add_special_tokens=False,
            max_length=self.body_max_len,
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.reranker_tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_model_len
        )
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        return inputs

    def _compute_scores(self, inputs, model) -> list[float]:
        """Score a batch by reading the assistant-position logits and computing
        ``P(yes) / (P(yes) + P(no))`` over the two token-id columns directly.

        Indexing the full vocab logits avoids the top-K logprobs truncation that
        affects ``generate``-style APIs when the reranker is highly confident
        and ``yes`` or ``no`` falls outside the returned top-K.
        """
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            tv = logits[:, self.yes_id]
            fv = logits[:, self.no_id]
            stacked = torch.stack([fv, tv], dim=1)
            stacked = torch.nn.functional.log_softmax(stacked, dim=1)
            return stacked[:, 1].exp().tolist()

    def _build_result(self, i: int, distance: float) -> SearchResult:
        return SearchResult(
            result=ResultData(
                module_name=self.data[i].get("module_name", []),
                kind=self.data[i].get("kind", ""),
                name=self.data[i].get("name", []),
                signature=self.data[i].get("signature", ""),
                type=self.data[i].get("type", ""),
                value=self.data[i].get("value"),
                docstring=self.data[i].get("docstring"),
                informal_name=self.data[i].get("informal_name"),
                informal_description=self.data[i].get("informal_description"),
            ),
            distance=float(distance),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_profile: bool = False,
        rerank: bool = True,
        retrieve_k: int | None = None,
    ) -> list[SearchResult] | tuple[list[SearchResult], dict]:
        """Retrieve the top `top_k` results for `query`.

        - `rerank=True` (default): cuVS-retrieve `retrieve_k` candidates
          (default `max(top_k * 2, 50)` when reranking), pass them through
          the reranker, return the top `top_k` by rerank score.
        - `rerank=False`: cuVS-retrieve `top_k` candidates directly and
          return them ordered by embedding distance (retriever-only).
        """
        profile: dict = {}

        t0 = time.time()
        q = f"Instruction: {self.INSTRUCTION}. Query: {query}"
        emb = self.embedding_model.encode([q], convert_to_numpy=True)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype("float32")
        profile["embedding_time"] = time.time() - t0

        if retrieve_k is None:
            retrieve_k = max(top_k * 2, 50) if rerank else top_k

        t0 = time.time()
        distances, indices = self._search_vectors(emb, retrieve_k)
        candidates = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i != -1]
        profile["retrieval_time"] = time.time() - t0
        profile["retrieval_backend"] = "torch_bruteforce"
        profile["num_candidates"] = len(candidates)
        profile["reranked"] = bool(rerank)

        if not candidates:
            if return_profile:
                profile["rerank_time"] = 0.0
                profile["total_time"] = profile["embedding_time"] + profile["retrieval_time"]
                return [], profile
            return []

        if not rerank:
            t0 = time.time()
            results = [self._build_result(i, d) for i, d in candidates[:top_k]]
            profile["postprocess_time"] = time.time() - t0
            profile["rerank_time"] = 0.0
            profile["total_time"] = (
                profile["embedding_time"] + profile["retrieval_time"] + profile["postprocess_time"]
            )
            if return_profile:
                return results, profile
            return results

        dist_map = {i: d for i, d in candidates}
        candidate_indices = [i for i, _ in candidates]
        n = len(candidate_indices)
        retriever_rank_scores = [(n - i) / n for i in range(n)]

        t0 = time.time()
        pairs = [self._format_pair(query, self.texts[i]) for i in candidate_indices]
        profile["prompt_build_time"] = time.time() - t0

        t0 = time.time()
        scores = self._rerank(pairs, fallback_scores=retriever_rank_scores)
        profile["rerank_time"] = time.time() - t0

        t0 = time.time()
        ranked = sorted(zip(candidate_indices, scores), key=lambda x: -x[1])
        results = [self._build_result(i, dist_map[i]) for i, _ in ranked[:top_k]]
        profile["postprocess_time"] = time.time() - t0

        profile["total_time"] = (
            profile["embedding_time"]
            + profile["retrieval_time"]
            + profile["prompt_build_time"]
            + profile["rerank_time"]
            + profile["postprocess_time"]
        )

        if return_profile:
            return results, profile
        return results

    def _rerank(
        self,
        pairs: list[str],
        fallback_scores: list[float],
    ) -> list[float]:
        """Score every (query, doc) pair across the reranker replicas.

        Pairs are evenly chunked across GPUs and scored in parallel through a
        ``ThreadPoolExecutor``. On unexpected failure the function returns the
        caller-supplied ``fallback_scores`` (rank-decreasing values in
        ``[0, 1]`` aligned with retriever order), so the worst case degrades
        to retriever-only ranking rather than a zeroed tie.
        """
        if not pairs:
            return []

        n = len(self.reranker_models)
        chunk = (len(pairs) + n - 1) // n
        chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]

        def _job(idx: int, payload: list[str]) -> list[float]:
            if not payload:
                return []
            inputs = self._process_inputs(payload, self.reranker_devices[idx])
            return self._compute_scores(inputs, self.reranker_models[idx])

        try:
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(_job, i, c) for i, c in enumerate(chunks) if i < n]
                chunk_scores = [f.result() for f in futs]
            return [s for cs in chunk_scores for s in cs]
        except Exception as e:
            print(f"Warning: rerank failed ({e}); falling back to retriever order")
            return list(fallback_scores)
