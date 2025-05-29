import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
from seismic import SeismicIndex
from backend import SparseBackend
from data import IndexResult


class SeismicBackend(SparseBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.index_path = os.path.join(self.index_dir, "cached")
        self._index = None

    def build(self, corpus_path: str):
        self._index = SeismicIndex.build(corpus_path)
        print("Number of documents:", self._index.len)
        print("Avg number of non-zero components:", self._index.nnz / self._index.len)
        print("Dimensionality of the vectors:", self._index.dim)
        self._index.print_space_usage_byte()
        self._index.save(self.index_path)

    def search(self, q: List[str], eq: Dict, ids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self._index is None:
            self.open()
        query_components = []
        query_values = []
        query_ids = []
        for qid, query in eq.items():
            query_ids.append(qid)
            query_components.append(np.array(list(query.keys()), dtype="U30"))
            query_values.append(np.array(list(query.values()), dtype=np.float32))
        search_results = self._index.batch_search(
            queries_ids=np.array(query_ids, dtype="U30"),
            query_components=query_components,
            query_values=query_values,
            k=top_k,
            query_cut=20,
            heap_factor=0.7,
            sorted=True,
            n_knn=0
        )
        results = defaultdict(list)
        for sr in search_results:
            for query_id, score, doc_id in sr:
                results[query_id].append(IndexResult(doc_id, score))
        return results

    def open(self):
        self._index = SeismicIndex.load(self.index_path + ".index.seismic")

    def close(self):
        if self._index is not None:
            self._index = None

    def exists(self) -> bool:
        return os.path.exists(self.index_path + ".index.seismic")
