import os
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import numpy as np
from joblib import dump, load
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import csr_array

from backend.base import SparseBackend
from data import IndexResult


def _as_postings(corpus):
    postings = corpus.T.tocsr()
    if max(postings.nnz, *postings.shape) <= np.iinfo(np.int32).max:
        postings.indptr = postings.indptr.astype(np.int32, copy=False)
        postings.indices = postings.indices.astype(np.int32, copy=False)
    return postings


@njit(nogil=True)
def _append_postings(batch_offsets, batch_docs, batch_weights, cursor, doc_indices, weights, doc_start):
    for dimension in range(cursor.size):
        start = batch_offsets[dimension]
        end = batch_offsets[dimension + 1]
        if start == end:
            continue
        dest = cursor[dimension]
        stop = dest + (end - start)
        doc_indices[dest:stop] = batch_docs[start:end] + doc_start
        weights[dest:stop] = batch_weights[start:end]
        cursor[dimension] = stop


def _build_postings(batches, scratch):
    ids, doc_starts, dtypes = [], [], set()
    counts = None
    for embeddings, batch_ids in batches:
        if len(batch_ids) != embeddings.shape[0]:
            raise ValueError("Sparse vector batch must have one document ID per row")
        batch = _as_postings(embeddings)
        if counts is None:
            counts = np.zeros(batch.shape[0], dtype=np.int64)
        if batch.shape[0] != counts.size:
            raise ValueError("All sparse vector batches must have the same dimension")
        counts += np.diff(batch.indptr)
        np.save(scratch, batch.indptr, allow_pickle=False)
        np.save(scratch, batch.indices, allow_pickle=False)
        np.save(scratch, batch.data, allow_pickle=False)
        dtypes.add(batch.dtype)
        doc_starts.append(len(ids))
        ids.extend(batch_ids)

    dimensions = 0 if counts is None else counts.size
    nnz = 0 if counts is None else int(counts.sum())
    index_dtype = np.int32 if max(dimensions, len(ids), nnz) <= np.iinfo(np.int32).max else np.int64
    offsets = np.zeros(dimensions + 1, dtype=index_dtype)
    if counts is not None:
        np.cumsum(counts, out=offsets[1:])
    cursor = offsets[:-1].copy()
    doc_indices = np.empty(nnz, dtype=index_dtype)
    weights = np.empty(nnz, dtype=np.result_type(*dtypes) if dtypes else np.float32)
    # The Python path also handles SciPy dtypes such as long double, which Numba cannot compile.
    append = _append_postings if weights.dtype in (np.float32, np.float64) else _append_postings.py_func
    scratch.seek(0)
    for doc_start in doc_starts:
        batch_offsets = np.load(scratch, allow_pickle=False)
        batch_docs = np.load(scratch, allow_pickle=False).astype(index_dtype, copy=False)
        batch_weights = np.load(scratch, allow_pickle=False).astype(weights.dtype, copy=False)
        append(batch_offsets, batch_docs, batch_weights, cursor, doc_indices, weights, doc_start)
    return csr_array((weights, doc_indices, offsets), shape=(dimensions, len(ids))), ids


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
        self._index = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / self.path.name
            with TemporaryFile(dir=tmp_dir) as scratch:
                postings, ids = _build_postings(batches, scratch)
            # Keep the document-by-dimension layout on disk, in CSC so reopening can reuse the arrays.
            dump((postings.T, ids), tmp_path, compress=0)
            os.replace(tmp_path, self.path)
        self._index = postings, ids

    def search(self, embeddings, qids, top_k):
        if self._index is None:
            corpus, ids = load(self.path)
            self._index = _as_postings(corpus), ids
            del corpus
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
