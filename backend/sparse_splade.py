import json
import logging
import os
from collections import defaultdict
from typing import List, Dict, Callable
from splade_index import SPLADE
from backend import SparseBackend
from data import IndexResult


class SpladeIndexWrapper:

    def __init__(self):
        self.splade: SPLADE = SPLADE()

    def build(self, corpus_path: str, encoder):
        docs = self._get_documents(corpus_path)
        self.splade.index(model=encoder,
                          documents=[doc['content'] for doc in docs],
                          document_ids=[doc['id'] for doc in docs])

    def _get_documents(self, corpus_path: str) -> List:
        assert os.path.exists(corpus_path), f'{corpus_path} file doesn\'t exist'
        _, file_ext = os.path.splitext(corpus_path)
        assert '.jsonl' == file_ext, 'No JSONL file found'
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return [json.loads(doc) for doc in f.readlines()]

    def save(self, index_path):
        logging.info(f"Saving index to {index_path}")
        self.splade.save(index_path)

    def retrieve(self, queries, k, sorted):
        return self.splade.retrieve(queries=queries, k=k, sorted=sorted)

    def close(self):
        self.splade = None


class SpladeIndexBackend(SparseBackend):

    def __init__(self, index_dir: str, encoder_provider: Callable):
        super().__init__(index_dir)
        self._encoder_provider = encoder_provider
        self._index = SpladeIndexWrapper()
        self.index_path = os.path.join(self.index_dir, "cached")

    def build(self, corpus_path: str):
        encoder = self._encoder_provider()
        self._index.build(encoder=encoder.model, corpus_path=corpus_path)
        self._index.save(self.index_path)

    def search(self, q: List[str], eq: Dict, ids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        search_results = self._index.retrieve(queries=q,k=top_k,sorted=True)
        results = defaultdict(list)
        for q_id, doc_ids, scores in zip(ids, search_results.doc_ids, search_results.scores):
            for doc_id, score in zip(doc_ids, scores):
                results[q_id].append(IndexResult(doc_id, float(score)))
        return results

    def close(self):
        self._index.close()

    def exists(self) -> bool:
        return os.path.exists(self.index_path)
