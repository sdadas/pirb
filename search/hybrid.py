import itertools
import json
import logging
import os.path
import random
import shutil
import tempfile
from abc import ABC
from collections import defaultdict
from typing import List, Iterable, Dict, Optional, Tuple, Union, Set, Callable
import numpy as np
import torch
import xgboost as xgb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel, \
    AutoModelForSeq2SeqLM
from data import IndexInput, IndexResult, RetrievalTask
from search.base import SearchIndex, SmartTemplate


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


class ClassifierReranker:

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.fp16 = kwargs.get("fp16", False)
        self.bf16 = kwargs.get("bf16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.template = kwargs.get("template", "{query}{sep}{sep}{passage}")
        model, tokenizer = self._load_classifier()
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.sep = self.tokenizer.sep_token
        assert self.sep is not None, "sep token is none"

    def _load_classifier(self):
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        dtype = torch.float32
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_name, torch_dtype=dtype).to(self.device)
        model.eval()
        return model, tokenizer

    def rerank(self, query: str, docs: List[str]):
        texts = [self.template.format(query=query, passage=doc, sep=self.sep) for doc in docs]
        tokens = self.tokenizer(texts, padding="longest", max_length=self.maxlen, truncation=True, return_tensors="pt")
        tokens.to(self.device)
        output = self.model(**tokens)
        logits = output.logits.detach().type(torch.float32).cpu().numpy()
        return np.squeeze(logits).tolist()


class Seq2SeqReranker:

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.fp16 = kwargs.get("fp16", False)
        self.bf16 = kwargs.get("bf16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.template = kwargs.get("template", "Query: {query} Document: {passage} Relevant:")
        self.yes_token = kwargs.get("yes_token", "yes")
        self.no_token = kwargs.get("no_token", "no")
        model, tokenizer = self._load_model()
        self.model = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        if isinstance(self.yes_token, int):
            self.yes_id = self.yes_token
        else:
            yes_ids = self.tokenizer.encode(self.yes_token, add_special_tokens=False, padding=False)
            assert len(yes_ids) == 1, yes_ids
            self.yes_id = yes_ids[0]
        if isinstance(self.no_token, int):
            self.no_id = self.no_token
        else:
            no_ids = self.tokenizer.encode(self.no_token, add_special_tokens=False, padding=False)
            assert len(no_ids) == 1, no_ids
            self.no_id = no_ids[0]
        self.smart_template = SmartTemplate(self.template, self.tokenizer, self.maxlen)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name, use_fast=False)
        dtype = torch.float32
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        model = AutoModelForSeq2SeqLM.from_pretrained(self.reranker_name, torch_dtype=dtype).to(self.device)
        model.eval()
        return model, tokenizer

    def rerank(self, query: str, docs: List[str]):
        inputs = self._use_naive_template(query, docs)
        inputs.to(self.device)
        outputs = self.model.generate(
            **inputs,
            num_beams=1,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )
        batch_scores = outputs.scores[0][:, [self.no_id, self.yes_id]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        batch_log_probs = batch_scores[:, 1].detach().tolist()
        return batch_log_probs

    def _use_smart_template(self, query: str, docs: List[str]):
        encoded = [{"input_ids": self.smart_template.encode(query=query, passage=doc)} for doc in docs]
        return self.tokenizer.pad(encoded, padding="longest", max_length=self.maxlen, return_tensors="pt")

    def _use_naive_template(self, query: str, docs: List[str]):
        texts = [self.template.format(query=query, passage=doc) for doc in docs]
        return self.tokenizer(texts, padding="longest", max_length=self.maxlen, truncation=True, return_tensors="pt")


class RerankerHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs
        self.rerank_limit = kwargs.get("rerank_limit", None)
        self.batch_size = kwargs.get("batch_size", 32)
        self.reranker = None

    def _load_reranker(self):
        reranker_type = self.args.get("reranker_type", "classifier")
        reranker_name = self.args['reranker_name']
        assert reranker_type in ("classifier", "seq2seq")
        logging.info(f"Loading {reranker_type} reranker {reranker_name}")
        if reranker_type == "classifier":
            self.reranker = ClassifierReranker(**self.args)
        elif reranker_type == "seq2seq":
            self.reranker = Seq2SeqReranker(**self.args)

    def merge_results(self, index_results: List[Dict], k: int) -> Dict:
        if self.reranker is None:
            self._load_reranker()
        results = {}
        queries, passages = self._load_task_data()
        for qid in tqdm(queries.keys(), desc="Reranking results"):
            rerank_func: Callable = self._rerank_limited if self.rerank_limit else self._rerank_all
            results[qid] = rerank_func(qid, queries, passages, index_results, k)
        return results

    def _rerank_all(self, qid: str, queries: Dict, passages: Dict, index_results: List[Dict], k: int):
        docids: Set = set()
        for index in index_results:
            hits = index.get(qid, [])
            docids.update({val.id for val in hits})
        return self._rerank(qid, list(docids), queries, passages, k)

    def _rerank_limited(self, qid: str, queries: Dict, passages: Dict, index_results: List[Dict], k: int):
        docids: Set = set()
        all_hits = []
        for index in index_results:
            hits = index.get(qid, [])
            all_hits.extend([hit for hit in hits if hit.id not in docids])
            docids.update({val.id for val in hits})
        all_docids = [hit.id for hit in all_hits]
        reranked_hits = self._rerank(qid, all_docids[:self.rerank_limit], queries, passages, k)
        results = []
        last_score = 0
        for idx in range(k):
            if idx >= len(all_hits):
                break
            if idx < self.rerank_limit:
                hit = reranked_hits[idx]
                results.append(hit)
                last_score = hit.score
            else:
                hit = all_hits[idx]
                results.append(IndexResult(hit.id, last_score - idx))
        return results

    def _rerank(self, qid: str, docids: List[str], queries: Dict, passages: Dict, k: int):
        query = queries[qid].text
        results = []
        for i in range(0, len(docids), self.batch_size):
            batch = docids[i:i + self.batch_size]
            docs = [passages[docid].text for docid in batch]
            with torch.no_grad():
                outputs = self.reranker.rerank(query, docs)
                outputs = [outputs] if not isinstance(outputs, list) else outputs
            results.extend([IndexResult(docid, float(score)) for docid, score in zip(batch, outputs)])
        results.sort(key=lambda v: -v.score)
        results = results[:k]
        return results

    def _load_task_data(self):
        logging.info("Loading queries for reranker")
        queries = {val.id: val for val in self.task.queries(self.data_dir)}
        logging.info("Loading passages for reranker")
        passages = {val.id: val for val in self.task.passages(self.data_dir)}
        return queries, passages

    def needs_cache(self) -> bool:
        return False

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
                 strategy: Union[HybridStrategy, Dict], task: RetrievalTask, rerank_limit: int):
        self.data_dir = data_dir
        self.hybrid_name = hybrid_name
        self.indices = indices
        self.strategy = strategy if isinstance(strategy, HybridStrategy) else self._load_strategy(strategy, rerank_limit)
        self.strategy.task = task
        self.strategy.data_dir = data_dir
        self.k0 = k0

    def _load_strategy(self, spec: Dict, rerank_limit: int):
        spec["rerank_limit"] = rerank_limit
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
