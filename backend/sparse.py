import os
from typing import List, Dict, Callable, Optional
from pyserini.search import LuceneImpactSearcher
from pyserini.encode import QueryEncoder
from backend.base import SparseBackend
from data import IndexResult


class LuceneBackend(SparseBackend):

    def __init__(self, index_dir: str, encoder_provider: Callable[[], QueryEncoder], threads: int = 8):
        super().__init__(index_dir)
        self.threads = threads
        self.encoder_provider = encoder_provider
        self.encoder = None
        self.searcher = None

    def build(self, corpus_path: str):
        from jnius import autoclass
        docs_dir = os.path.dirname(corpus_path)
        args = [
            "-collection", "JsonVectorCollection",
            "-input", os.path.abspath(docs_dir),
            "-index", os.path.abspath(self.index_dir),
            "-threads", str(self.threads),
            "-generator", "DefaultLuceneDocumentGenerator",
            "-impact",
            "-pretokenized",
            "-storeDocvectors"
        ]
        JIndexCollection = autoclass('io.anserini.index.IndexCollection')
        JIndexCollection.main(args)

    def search(self, q: List[str], eq: Dict, ids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self.searcher is None:
            self.open()
        batch_results = self.searcher.batch_search(q, ids, k=top_k, threads=self.threads)
        for qid, relevant in batch_results.items():
            batch_results[qid] = [IndexResult(val.docid, val.score) for val in relevant]
        return batch_results

    def open(self):
        self.encoder = self.encoder_provider()
        self.searcher = LuceneImpactSearcher(self.index_dir, self.encoder)

    def exists(self) -> bool:
        return os.path.exists(self.index_dir)
