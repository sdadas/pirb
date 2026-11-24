import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from joblib import dump, load
from scipy.sparse import csr_array, vstack

from backend.base import SparseBackend
from data import IndexResult


class SparseCSRBackend(SparseBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.path = Path(index_dir) / "vectors.joblib"
        self._index = None

    def build(self, batches):
        vectors, ids = [], []
        for embeddings, batch_ids in batches:
            vectors.append(embeddings)
            ids.extend(batch_ids)
        corpus = vstack(vectors, format="csr") if vectors else csr_array((0, 0), dtype=np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / self.path.name
            dump((corpus, ids), tmp_path)
            os.replace(tmp_path, self.path)
        self._index = corpus.T.tocsr(), ids

    def search(self, embeddings, qids, top_k):
        if self._index is None:
            corpus, ids = load(self.path)
            self._index = corpus.T.tocsr(), ids
        corpus_t, ids = self._index
        k = min(top_k, len(ids))
        if k <= 0:
            return {qid: [] for qid in qids}

        results = {}
        score_batch_size = 32
        for start in range(0, len(qids), score_batch_size):
            end = start + score_batch_size
            scores = (embeddings[start:end] @ corpus_t).toarray()
            rankings = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for qid, row, ranking in zip(qids[start:end], scores, rankings):
                results[qid] = [IndexResult(ids[i], float(row[i])) for i in ranking]
        return results

    def exists(self):
        return self.path.is_file()

    def close(self):
        self._index = None
