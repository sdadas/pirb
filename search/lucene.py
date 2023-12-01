import json
import logging
import os.path
import shutil
from typing import List, Iterable, Dict
from pyserini.search import LuceneSearcher
from jnius import autoclass
from tqdm import tqdm
from data import IndexInput, IndexResult
from search.base import SearchIndex


class LuceneIndex(SearchIndex):

    def __init__(self, data_dir: str, lang: str, threads: int):
        self.data_dir = data_dir
        self.index_dir = os.path.join(self.data_dir, "lucene_index")
        self.lang = lang
        self.searcher = None
        self.threads = threads

    def exists(self) -> bool:
        return os.path.exists(self.index_dir)

    def build(self, docs: Iterable[IndexInput]):
        if self.exists(): shutil.rmtree(self.index_dir)
        docs_dir = os.path.join(self.data_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        with open(os.path.join(docs_dir, "passages.jsonl"), "w", encoding="utf-8") as docs_file:
            for doc in docs:
                json_value = {"id": doc.id, "contents": doc.text}
                docs_file.write(json.dumps(json_value, ensure_ascii=False))
                docs_file.write("\n")
        logging.info("Building lucene index %s", self.index_dir)
        args = [
            "-collection", "JsonCollection",
            "-language", self.lang,
            "-input", os.path.abspath(docs_dir),
            "-index", os.path.abspath(self.index_dir),
            "-threads", str(self.threads),
            "-generator", "DefaultLuceneDocumentGenerator",
            "-storePositions",
            "-storeDocvectors",
            "-storeRaw",
            "-bm25.accurate"
        ]
        JIndexCollection = autoclass('io.anserini.index.IndexCollection')
        JIndexCollection.main(args)

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None, overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self.searcher is None: self._open_searcher()
        results = {}
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching in lucene index", disable=not verbose):
            batch: List[IndexInput] = queries[i:i + batch_size]
            texts = [val.text for val in batch]
            ids = [val.id for val in batch]
            batch_results = self.searcher.batch_search(texts, ids, k=k, threads=self.threads)
            for qid, relevant in batch_results.items():
                batch_results[qid] = [IndexResult(val.docid, val.score) for val in relevant]
            results.update(batch_results)
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def _open_searcher(self):
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_language(self.lang)
        self.searcher = searcher

    def name(self):
        return "sparse"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return {"name": "bm25"}