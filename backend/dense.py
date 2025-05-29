import os
from collections import defaultdict
from typing import List, Dict
import torch
from joblib import dump, load
import numpy as np
from sentence_transformers.util import semantic_search
from backend.base import DenseBackend
from data import IndexResult


class FaissBackend(DenseBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.index_vectors_path = os.path.join(self.index_dir, "index.bin")
        self.index_ids_path = os.path.join(self.index_dir, "ids.bin")
        self._index = None
        self._ids = None

    def build(self, embeddings, ids: List[str]):
        import faiss
        embeddings = np.vstack(embeddings)
        dim = embeddings.shape[1]
        res = faiss.IndexFlatIP(dim)
        res.add(embeddings)
        faiss.write_index(res, self.index_vectors_path)
        dump(ids, self.index_ids_path)
        self._index = res
        self._ids = ids

    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self._index is None or self._ids is None:
            self.open()
        results = defaultdict(list)
        sim, indices = self._index.search(embeddings, top_k)
        sim = sim.tolist()
        indices = indices.tolist()
        for i, qid in enumerate(qids):
            for k in range(top_k):
                idx = indices[i][k]
                docid = self._ids[idx]
                score = sim[i][k]
                results[qid].append(IndexResult(docid, score))
        return results

    def open(self):
        import faiss
        self._index = faiss.read_index(self.index_vectors_path)
        self._ids = load(self.index_ids_path)

    def exists(self) -> bool:
        return os.path.exists(self.index_vectors_path) and os.path.exists(self.index_ids_path)


class RawVectorsBackend(DenseBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.index_vectors_path = os.path.join(self.index_dir, "index.bin")
        self.index_ids_path = os.path.join(self.index_dir, "ids.bin")
        self._index = None
        self._ids = None

    def build(self, embeddings, ids: List[str]):
        res = torch.vstack(embeddings)
        torch.save(res, self.index_vectors_path)
        dump(ids, self.index_ids_path)
        self._index = res.float()
        self._ids = ids

    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        embeddings = embeddings.detach().cpu().float()
        results = defaultdict(list)
        hits = semantic_search(embeddings, self._index, query_chunk_size=128, top_k=top_k)
        for i, qid in enumerate(qids):
            query_hits = hits[i]
            for k in range(top_k):
                hit = query_hits[k]
                docid = self._ids[int(hit["corpus_id"])]
                score = hit["score"]
                results[qid].append(IndexResult(docid, score))
        return results

    def open(self):
        self._index = torch.load(self.index_vectors_path).float()
        self._ids = load(self.index_ids_path)

    def exists(self) -> bool:
        return os.path.exists(self.index_vectors_path) and os.path.exists(self.index_ids_path)

    def supports_tensors(self):
        return True
