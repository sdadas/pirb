import os
import shutil
from typing import Iterable, List, Dict, Optional
import bm25s
from tqdm import tqdm
from data import IndexInput, IndexResult
from search import SearchIndex
from joblib import dump, load
from utils.stemming import get_stopwords, get_lemmatizer


class BM25Index(SearchIndex):

    def __init__(self, data_dir: str, lang: str):
        self.data_dir = data_dir
        self.index_dir = os.path.join(data_dir, "bm25_index")
        self.lang = lang
        self.stopwords = get_stopwords(lang)
        self.lemmatizer = get_lemmatizer(lang)
        self.retriever: Optional[bm25s.BM25] = None
        self.ids: Optional[List[str]] = None

    def exists(self) -> bool:
        return os.path.exists(self.index_dir)

    def build(self, docs: Iterable[IndexInput]):
        if self.exists():
            shutil.rmtree(self.index_dir)
        texts, ids = [], []
        for doc in docs:
            texts.append(doc.text)
            ids.append(doc.id)
        tokens = bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.lemmatizer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens)
        retriever.save(self.index_dir)
        dump(ids, os.path.join(self.index_dir, "ids.bin"))
        self.retriever = retriever
        self.ids = ids

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None, overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self.retriever is None:
            self.retriever = bm25s.BM25.load(self.index_dir)
            self.ids = load(os.path.join(self.index_dir, "ids.bin"))
        results = {}
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching in BM25 index", disable=not verbose):
            batch: List[IndexInput] = queries[i:i + batch_size]
            texts = [val.text for val in batch]
            query_tokens = bm25s.tokenize(texts, stemmer=self.lemmatizer, show_progress=False)
            r = self.retriever.retrieve(query_tokens, k=k, sorted=True, show_progress=False)
            for idx, query in enumerate(batch):
                query_results = [
                    IndexResult(self.ids[r.documents[idx][i]], float(r.scores[idx][i]))
                    for i in range(k)
                ]
                results[query.id] = query_results
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def name(self):
        return "bm25"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return {"name": "bm25"}

    def index_path(self) -> str:
        return self.index_dir
