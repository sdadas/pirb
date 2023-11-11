import json
import logging
import os.path
import random
import shutil
import tempfile
from abc import ABC
from collections import defaultdict
from typing import List, Iterable, Dict, Optional, Tuple, Union, Set
import numpy as np
import torch
import xgboost as xgb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from data import IndexInput, IndexResult, RetrievalTask
from search.base import SearchIndex


class HybridStrategy(ABC):

    def __init__(self, **kwargs):
        self.task: Optional[RetrievalTask] = None
        self.data_dir: Optional[str] = None

    def merge_results(self, index_results: List[Dict], k: int) -> Dict:
        raise NotImplementedError()

    def model_dict(self) -> Dict:
        raise NotImplementedError()

    def needs_cache(self) -> bool:
        return False


class WeightedHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = kwargs["weights"]
        assert isinstance(self.weights, list)

    def merge_results(self, index_results: List[Dict], k: int) -> Dict:
        qids = set()
        for index in index_results:
            qids.update(index.keys())
        results = {}
        for qid in qids:
            combined_map = defaultdict(float)
            for i, index in enumerate(index_results):
                hits = index.get(qid, [])
                for hit in hits:
                    combined_map[hit.id] += self.weights[i] * hit.score
            combined = [IndexResult(key, val) for key, val in combined_map.items()]
            combined.sort(key=lambda v: -v.score)
            combined = combined[:k]
            results[qid] = combined
        return results

    def model_dict(self) -> Dict:
        return {"type": "weighted", "weights": self.weights}


class RerankerHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.reranker_type = kwargs.get("reranker_type", "classifier")
        self.batch_size = kwargs.get("batch_size", 32)
        self.fp16 = kwargs.get("fp16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.template = kwargs.get("template", "{query}</s>{passage}")
        self.args = kwargs
        assert self.reranker_type in ("classifier",)
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model = None

    def _load_classifier(self):
        logging.info(f"Loading reranker {self.reranker_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_name).to(self.device)
        model.eval()
        if self.fp16: model.half()
        self.model = model
        self.tokenizer = tokenizer

    def merge_results(self, index_results: List[Dict], k: int) -> Dict:
        if self.model is None or self.tokenizer is None:
            self._load_classifier()
        results = {}
        queries, passages = self._load_task_data()
        for qid in tqdm(queries.keys(), desc="Reranking results"):
            docids: Set = set()
            for index in index_results:
                hits = index.get(qid, [])
                docids.update({val.id for val in hits})
            hits = self._rerank(qid, list(docids), queries, passages, k)
            results[qid] = hits
        return results

    def _rerank(self, qid: str, docids: List[str], queries: Dict, passages: Dict, k: int):
        query = queries[qid].text
        results = []
        eos = self.tokenizer.eos_token
        assert eos is not None, "eos token is none"
        for i in range(0, len(docids), self.batch_size):
            batch = docids[i:i + self.batch_size]
            texts = [self.template.format(query=query, passage=passages[docid].text, eos=eos) for docid in batch]
            with torch.no_grad():
                outputs = self._rerank_batch(texts)
                outputs = [outputs] if not isinstance(outputs, list) else outputs
            results.extend([IndexResult(docid, float(score)) for docid, score in zip(batch, outputs)])
        results.sort(key=lambda v: -v.score)
        results = results[:k]
        return results

    def _rerank_batch(self, texts: List[str]):
        tokens = self.tokenizer(texts, padding=True, max_length=self.maxlen, truncation=True, return_tensors="pt")
        tokens.to(self.device)
        output = self.model(**tokens)
        logits = output.logits.detach().cpu().numpy()
        return np.squeeze(logits).tolist()

    def _load_task_data(self):
        logging.info("Loading queries for reranker")
        queries = {val.id: val for val in self.task.queries(self.data_dir)}
        logging.info("Loading passages for reranker")
        passages = {val.id: val for val in self.task.passages(self.data_dir)}
        return queries, passages

    def needs_cache(self) -> bool:
        return True

    def model_dict(self) -> Dict:
        return {"type": "reranker", **self.args}


class XGBRankerHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.serialized_model = kwargs.get("model", None)
        self.model = self._load_model() if self.serialized_model else None

    def _load_model(self):
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = os.path.join(temp_dir, "model.json")
            with open(temp_path, "w", encoding="utf8") as model_file:
                model_file.write(json.dumps(self.serialized_model))
            model = xgb.XGBRanker()
            model.load_model(temp_path)
            logging.info("Loaded XGBRankerHybrid model")
            return model
        finally:
            shutil.rmtree(temp_dir)

    def fit(self, queries: List[IndexInput], index_results: List[Dict], valid_fraction: float = 0.01):
        logging.info("Training new XGBRankerHybrid")
        grouped = [self._build_training_data(query, index_results) for query in queries]
        random.shuffle(grouped)
        valid_num = int(len(queries) * valid_fraction)
        train = grouped[:-valid_num] if valid_fraction > 0 else grouped
        valid = grouped[-valid_num:] if valid_fraction > 0 else None
        X_train = np.vstack([x for x, y, g in train])
        y_train = np.hstack([y for x, y, g in train])
        groups = [g for x, y, g in train]
        model = self._train_model(X_train, y_train, groups)
        if valid is not None:
            self._validate_model(model, valid)
        self.model = model

    def _build_training_data(self, query: IndexInput, index_results: List[Dict]):
        relevant = {docid: query.relevant_scores[idx] for idx, docid in enumerate(query.relevant)}
        return self._build_data_points(query.id, relevant, index_results)

    def _build_data_points(self, query_id: str, relevant_scores: Dict, index_results: List[Dict], return_docids=False):
        all_hits = [index.get(query_id, []) for index in index_results]
        all_hits = [{hit.id: hit.score for hit in hits} for hits in all_hits]
        all_docs = set()
        for hits in all_hits:
            all_docs.update(hits.keys())
        all_docs = sorted(list(all_docs))
        relevance = []
        max_scores = [(max(hits.values()) if len(hits) > 0 else 0.0) for hits in all_hits]
        min_scores = [(min(hits.values()) if len(hits) > 0 else 0.0) for hits in all_hits]
        points = []
        for docid in all_docs:
            points.append(self._build_data_point(docid, all_hits, max_scores, min_scores))
            relevance.append(int(relevant_scores.get(docid, 0)))
        if return_docids:
            return np.vstack(points), np.array(relevance), len(points), all_docs
        else:
            return np.vstack(points), np.array(relevance), len(points)

    def _build_data_point(self, docid: str, all_hits: List[Dict], max_scores: List, min_scores: List):
        point = []
        for idx, hits in enumerate(all_hits):
            point.append(hits.get(docid, 0.0))
            point.append(1 if docid in hits else 0)
            point.append(max_scores[idx])
            point.append(min_scores[idx])
        return np.array(point)

    def _train_model(self, X_train, y_train, groups):
        model = xgb.XGBRanker(
            tree_method="hist",
            device="cuda",
            booster="gbtree",
            objective="rank:pairwise",
            random_state=42,
            learning_rate=0.1,
            colsample_bytree=0.9,
            eta=0.05,
            max_depth=6,
            n_estimators=100,
            subsample=0.75
        )
        model.fit(X_train, y_train, group=groups, verbose=True)
        return model

    def _validate_model(self, model: xgb.XGBRanker, grouped: List[Tuple]):
        num_correct = 0
        for idx, group in enumerate(grouped):
            x, y, _ = group
            pred = model.predict(x)
            pred_idx = int(np.argmax(pred))
            if y[pred_idx] > 0:
                num_correct += 1
        print(f"Accuracy: {100 * num_correct / len(grouped):.4f}%")

    def _get_serialized_model(self):
        if self.serialized_model is None and self.model is not None:
            temp_dir = tempfile.mkdtemp()
            try:
                temp_path = os.path.join(temp_dir, "model.json")
                self.model.save_model(temp_path)
                with open(temp_path, "r", encoding="utf-8") as model_file:
                    self.serialized_model = json.load(model_file)
            finally:
                shutil.rmtree(temp_dir)
        return self.serialized_model

    def merge_results(self, index_results: List[Dict], k: int) -> Dict:
        assert self.model is not None, "model is null"
        qids = set()
        for index in index_results:
            qids.update(index.keys())
        empty_relevant = {}
        results = {}
        for qid in tqdm(qids, desc=f"Merging results from {len(index_results)} indices"):
            x, y, g, docids = self._build_data_points(qid, empty_relevant, index_results, return_docids=True)
            output = self.model.predict(x)
            combined = [IndexResult(docid, score) for docid, score in zip(docids, output)]
            combined.sort(key=lambda v: -v.score)
            combined = combined[:k]
            results[qid] = combined
        return results

    def model_dict(self) -> Dict:
        return {"type": "xbgranker", "model": self._get_serialized_model()}


class HybridIndex(SearchIndex):

    def __init__(self, data_dir: str, hybrid_name: str, indices: List[SearchIndex], k0: int,
                 strategy: Union[HybridStrategy, Dict], task: RetrievalTask):
        self.data_dir = data_dir
        self.hybrid_name = hybrid_name
        self.indices = indices
        self.strategy = strategy if isinstance(strategy, HybridStrategy) else self._load_strategy(strategy)
        self.strategy.task = task
        self.strategy.data_dir = data_dir
        self.k0 = k0

    def _load_strategy(self, spec: Dict):
        strategy_type = spec["type"]
        types = {"weighted": WeightedHybrid, "xbgranker": XGBRankerHybrid, "reranker": RerankerHybrid}
        cls = types[strategy_type]
        return cls(**spec)

    def exists(self) -> bool:
        return all((index.exists() for index in self.indices))

    def build(self, docs: Iterable[IndexInput]):
        for index in self.indices:
            if not index.exists():
                index.build(docs)

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None,
               overwrite=False) -> Dict:
        needs_cache = not overwrite and self.strategy.needs_cache()
        results = self.load_results_from_cache(k, cache_prefix) if needs_cache else None
        if results is not None: return results
        index_results = []
        for index in self.indices:
            results = index.search(
                queries, self.k0, batch_size, verbose=verbose, cache_prefix=cache_prefix, overwrite=overwrite
            )
            index_results.append(results)
        results = self.strategy.merge_results(index_results, k)
        if self.strategy.needs_cache():
            self.save_results_to_cache(k, cache_prefix, results)
        return results

    def results_exist(self, k: int, cache_prefix: str):
        for index in self.indices:
            if not index.results_exist(k, cache_prefix):
                return False
        return True

    def name(self):
        return f"hybrid_{self.hybrid_name}"

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.indices[0])

    def basedir(self):
        return getattr(self.indices[0], "data_dir")

    def model_dict(self) -> Dict:
        return {
            "name": self.hybrid_name,
            "type": "hybrid",
            "models": [index.model_dict() for index in self.indices],
            "k0": self.k0,
            "strategy": self.strategy.model_dict()
        }
