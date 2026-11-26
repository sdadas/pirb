import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from joblib import dump, load
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import csr_array, vstack

from backend.base import SparseBackend
from data import IndexResult


@njit(parallel=True, nogil=True)
def _accumulate_scores(posting_offsets, doc_indices, doc_weights,
                       query_offsets, dimensions, query_weights, num_docs):
    scores = np.empty((query_offsets.size - 1, num_docs), dtype=np.float32)
    for query in prange(scores.shape[0]):
        row = scores[query]
        row.fill(0)
        for entry in range(query_offsets[query], query_offsets[query + 1]):
            dimension = dimensions[entry]
            weight = query_weights[entry]
            for pos in range(posting_offsets[dimension], posting_offsets[dimension + 1]):
                row[doc_indices[pos]] += weight * doc_weights[pos]
    return scores


def _stable_topk(row, k):
    keys = -row
    if k == keys.size:
        return np.argsort(keys, kind="stable")
    cutoff = np.partition(keys, k - 1)[k - 1]
    if np.isnan(cutoff):
        return np.argsort(keys, kind="stable")[:k]
    better = np.flatnonzero(keys < cutoff)
    # Keep the original document order when a tie straddles the cutoff.
    tied = np.flatnonzero(keys == cutoff)[:k - better.size]
    candidates = np.concatenate((better, tied))
    return candidates[np.lexsort((candidates, keys[candidates]))]


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
            dump((corpus, ids), tmp_path, compress=3)
            index = self._prepare_index(corpus, ids)
            os.replace(tmp_path, self.path)
        self._index = index

    @staticmethod
    def _prepare_index(corpus, ids):
        postings = corpus.T.tocsr()
        if postings.dtype == np.float32 and max(postings.nnz, *postings.shape) <= np.iinfo(np.int32).max:
            postings.indptr = postings.indptr.astype(np.int32, copy=False)
            postings.indices = postings.indices.astype(np.int32, copy=False)
        return postings, ids

    def search(self, embeddings, qids, top_k):
        if self._index is None:
            corpus, ids = load(self.path)
            self._index = self._prepare_index(corpus, ids)
        corpus_t, ids = self._index
        k = min(top_k, len(ids))
        if k <= 0:
            return {qid: [] for qid in qids}
        if len(qids) == 0:
            return {}
        if embeddings.shape[1] != corpus_t.shape[0]:
            raise ValueError("Query and document vectors must have the same dimension")

        use_numba = embeddings.format == "csr" and embeddings.dtype == corpus_t.dtype == np.float32
        threads = get_num_threads() if use_numba else 0
        search_threads = min(threads, 16, len(qids))
        if search_threads < threads:
            set_num_threads(search_threads)
        results = {}
        score_batch_size = 32
        try:
            for start in range(0, len(qids), score_batch_size):
                end = start + score_batch_size
                queries = embeddings[start:end]
                if use_numba:
                    scores = _accumulate_scores(corpus_t.indptr, corpus_t.indices, corpus_t.data,
                                                queries.indptr, queries.indices, queries.data, len(ids))
                    rankings = (_stable_topk(row, k) for row in scores)
                else:
                    scores = (queries @ corpus_t).toarray()
                    rankings = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                for qid, row, ranking in zip(qids[start:end], scores, rankings):
                    results[qid] = [IndexResult(ids[i], float(row[i])) for i in ranking]
        finally:
            if search_threads < threads:
                set_num_threads(threads)
        return results

    def exists(self):
        return self.path.is_file()

    def close(self):
        self._index = None
