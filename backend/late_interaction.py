import os
from collections import defaultdict

from joblib import dump, load
from typing import List, Dict

from pylate import indexes, retrieve

from backend.base import LateInteractionBackend
from data import IndexResult


class PlaidBackend(LateInteractionBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.index_dir = index_dir
        self.index_path = os.path.join(self.index_dir, "index.bin")
        self._index = None
        self._ids = None

    def build(self, embeddings, ids: List[str]):
        self._index = indexes.PLAID(
            index_folder=self.index_dir,
            index_name="index.bin"
        )
        self._index.add_documents(
            documents_ids=ids,
            documents_embeddings=embeddings,
        )

    def open(self):
        self._index = indexes.PLAID(
            index_folder=self.index_dir,
            index_name="index.bin"
        )

    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self._index is None:
            self.open()
        retriever = retrieve.ColBERT(index=self._index)
        hits = retriever.retrieve(queries_embeddings=embeddings, k=top_k)
        results = defaultdict(list)
        for i, qid in enumerate(qids):
            query_hits = hits[i]
            for k in range(top_k):
                hit = query_hits[k]
                docid = hit["id"]
                score = hit["score"]
                results[qid].append(IndexResult(docid, score))
        return results

    def exists(self) -> bool:
        return os.path.exists(self.index_path)


class VoyagerBackend(LateInteractionBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        self.index_path = os.path.join(self.index_dir, "index.bin")
        self._index = None
        self._ids = None

    def build(self, embeddings, ids: List[str]):
        self._index = indexes.Voyager()
        self._index.add_documents(
            documents_ids=ids,
            documents_embeddings=embeddings,
        )
        dump(self._index, self.index_path)

    def open(self):
        self._index = load(self.index_path)

    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self._index is None:
            self.open()
        retriever = retrieve.ColBERT(index=self._index)
        hits = retriever.retrieve(queries_embeddings=embeddings, k=top_k)
        results = defaultdict(list)
        for i, qid in enumerate(qids):
            query_hits = hits[i]
            for k in range(top_k):
                hit = query_hits[k]
                docid = hit["id"]
                score = hit["score"]
                results[qid].append(IndexResult(docid, score))
        return results

    def exists(self) -> bool:
        return os.path.exists(self.index_path)
