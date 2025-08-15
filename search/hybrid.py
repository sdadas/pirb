import json
import logging
import os.path
import random
import shutil
import tempfile
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Iterable, Dict, Optional, Tuple, Union, Set, Callable
import numpy as np
from scipy.special import expit
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel, \
    AutoModelForSeq2SeqLM
from data import IndexInput, IndexResult, RetrievalTask
from search.base import SearchIndex, SmartTemplate
from utils.system import install_package


class HybridStrategy(ABC):

    def __init__(self, **kwargs):
        self.task: Optional[RetrievalTask] = None
        self.data_dir: Optional[str] = None

    @abstractmethod
    def merge_results(self, queries: List[IndexInput], index_results: List[Dict], k: int) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def model_dict(self) -> Dict:
        raise NotImplementedError()

    def needs_cache(self) -> bool:
        return False


class Reranker(ABC):

    @abstractmethod
    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        raise NotImplementedError()

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        return self.rerank_pairs([query] * len(docs), docs, proba)


class WeightedHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = kwargs["weights"]
        assert isinstance(self.weights, list)

    def merge_results(self, queries: List[IndexInput], index_results: List[Dict], k: int) -> Dict:
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


class ClassifierReranker(Reranker):

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.fp16 = kwargs.get("fp16", False)
        self.bf16 = kwargs.get("bf16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.template = kwargs.get("template", "{query}{sep}{sep}{passage}")
        self.use_bettertransformer = kwargs.get("use_bettertransformer", False)
        self.model_kwargs = kwargs.get("model_kwargs", {})
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
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_name, torch_dtype=dtype, **self.model_kwargs
        ).to(self.device)
        model.eval()
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer
            try:
                model = BetterTransformer.transform(model)
            except NotImplementedError:
                logging.warning(f"Model {model.config.model_type} not supported in BetterTransformer")
        return model, tokenizer

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        assert len(queries) == len(docs)
        texts = [self.template.format(query=queries[i], passage=docs[i], sep=self.sep) for i in range(len(docs))]
        tokens = self.tokenizer(texts, padding="longest", max_length=self.maxlen, truncation=True, return_tensors="pt")
        tokens.to(self.device)
        output = self.model(**tokens)
        logits = output.logits.detach().type(torch.float32).cpu().numpy()
        logits = np.squeeze(logits)
        return expit(logits).tolist() if proba else logits.tolist()


class Seq2SeqReranker(Reranker):

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.fp16 = kwargs.get("fp16", False)
        self.bf16 = kwargs.get("bf16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.template = kwargs.get("template", "Query: {query} Document: {passage} Relevant:")
        self.yes_token = kwargs.get("yes_token", "yes")
        self.no_token = kwargs.get("no_token", "no")
        self.use_bettertransformer = kwargs.get("use_bettertransformer", False)
        self.model_kwargs = kwargs.get("model_kwargs", {})
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
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.reranker_name, torch_dtype=dtype, **self.model_kwargs
        ).to(self.device)
        model.eval()
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer
            try:
                model = BetterTransformer.transform(model)
            except NotImplementedError:
                logging.warning(f"Model {model.config.model_type} not supported in BetterTransformer")
        return model, tokenizer

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        assert len(queries) == len(docs)
        inputs = self._use_naive_template(queries, docs)
        inputs.to(self.device)
        outputs = self.model.generate(
            **inputs,
            num_beams=1,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )
        batch_scores = outputs.scores[0][:, [self.no_id, self.yes_id]]
        if proba:
            batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
        else:
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        batch_probs = batch_scores[:, 1].detach().tolist()
        return batch_probs

    def _use_smart_template(self, queries: List[str], docs: List[str]):
        num_docs = range(len(docs))
        encoded = [{"input_ids": self.smart_template.encode(query=queries[i], passage=docs[i])} for i in num_docs]
        return self.tokenizer.pad(encoded, padding="longest", max_length=self.maxlen, return_tensors="pt")

    def _use_naive_template(self, queries: List[str], docs: List[str]):
        texts = [self.template.format(query=queries[i], passage=docs[i]) for i in range(len(docs))]
        return self.tokenizer(texts, padding="longest", max_length=self.maxlen, truncation=True, return_tensors="pt")


class Qwen3Reranker(Reranker):

    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.torch_dtype = torch.float32
        if kwargs.get("bf16", False):
            self.torch_dtype = torch.bfloat16
        elif kwargs.get("fp16", False):
            self.torch_dtype = torch.float16
        self.max_length = kwargs.get("max_seq_length", 8192)
        self.prompt = kwargs.get("prompt", None)
        if self.prompt is not None:
            logging.info(f"Using custom prompt: '{self.prompt}'")
        else:
            self.prompt = "Given a web search query, retrieve relevant passages that answer the query"
        model, tokenizer = self._load_model()
        self.model = model
        self.tokenizer = tokenizer
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")
        prefix_tokens, suffix_tokens = self._get_prefix_suffix()
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens

    def _get_prefix_suffix(self):
        system_msg = ("Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                      "Note that the answer can only be \"yes\" or \"no\".")
        prefix = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        return prefix_tokens, suffix_tokens

    def _load_model(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.reranker_name, torch_dtype=self.torch_dtype, attn_implementation="flash_attention_2", device_map="cuda"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name, padding_side="left")
        return model, tokenizer

    def format_instruction(self, query: str, doc: str):
        return f"<Instruct>: {self.prompt}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(
            pairs, padding=False, truncation="longest_first",
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def compute_logits(self, inputs):
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        pairs = [self.format_instruction(query, doc) for query, doc in zip(queries, docs)]
        inputs = self.process_inputs(pairs)
        return self.compute_logits(inputs)


class FlagReranker(Reranker):

    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.use_fp16 = kwargs.get("fp16", False)
        self.use_bf16 = kwargs.get("bf16", False)
        self.reranker_type = kwargs["reranker_type"]
        self.cutoff_layers = kwargs.get("cutoff_layers", 28)
        self.compress_ratio = kwargs.get("compress_ratio", None)
        self.compress_layers = kwargs.get("compress_layers", None)
        self.max_length = kwargs.get("max_seq_length", None)
        self.prompt = kwargs.get("prompt", None)
        assert self.reranker_type in ("flag_classifier", "flag_llm", "flag_layerwise_llm", "flag_lightweight_llm")
        self.model = self._load_model()
        if self.prompt is not None:
            logging.info(f"Using custom prompt: '{self.prompt}'")

    def _load_model(self):
        if self.reranker_type == "flag_classifier":
            from FlagEmbedding.flag_reranker import FlagReranker
            model = FlagReranker(self.reranker_name, use_fp16=self.use_fp16 or self.use_bf16)
        elif self.reranker_type == "flag_llm":
            from FlagEmbedding.flag_reranker import FlagLLMReranker
            model = FlagLLMReranker(self.reranker_name, use_fp16=self.use_fp16, use_bf16=self.use_bf16)
        elif self.reranker_type == "flag_layerwise_llm":
            from FlagEmbedding.flag_reranker import LayerWiseFlagLLMReranker
            model = LayerWiseFlagLLMReranker(self.reranker_name, use_fp16=self.use_fp16, use_bf16=self.use_bf16)
        elif self.reranker_type == "flag_lightweight_llm":
            from FlagEmbedding.flag_reranker import LightWeightFlagLLMReranker
            model = LightWeightFlagLLMReranker(self.reranker_name, use_fp16=self.use_fp16, use_bf16=self.use_bf16)
        else:
            raise ValueError(f"Unknown reranker type {self.reranker_type}")
        return model

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        pairs = list(zip(queries, docs))
        args = {"normalize": proba, "batch_size": len(pairs)}
        if self.max_length is not None:
            args["max_length"] = self.max_length
        if self.reranker_type in ("flag_layerwise_llm", "flag_lightweight_llm"):
            args["cutoff_layers"] = self.cutoff_layers
        if self.reranker_type == "flag_lightweight_llm":
            if self.compress_ratio is not None:
                args["compress_ratio"] = self.compress_ratio
            if self.compress_layers is not None:
                args["compress_layers"] = self.compress_layers
            if self.prompt is not None:
                args["prompt"] = self.prompt
        res = np.array(self.model.compute_score(pairs, **args)).flatten().tolist()
        assert len(res) == len(pairs)
        return res


class JinaReranker(Reranker):

    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.use_bf16 = kwargs.get("bf16", False)
        self.use_fp16 = kwargs.get("fp16", False)
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_seq_length", 1024)
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.model = self._load_model()

    def _load_model(self):
        dtype = "auto"
        if self.use_bf16:
            dtype = torch.bfloat16
        elif self.use_fp16:
            dtype = torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            **self.model_kwargs
        )
        model.to("cuda")
        model.eval()
        return model

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        pairs = list(zip(queries, docs))
        return self.model.compute_score(pairs, batch_size=self.batch_size, max_length=self.max_length)


class PylateReranker(Reranker):

    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.colbert_kwargs = kwargs.get("colbert_kwargs", {})
        self.batch_size = kwargs.get("batch_size", 32)
        self.torch_dtype = torch.float32
        if kwargs.get("bf16", False):
            self.torch_dtype = torch.bfloat16
        elif kwargs.get("fp16", False):
            self.torch_dtype = torch.float16
        self.model_kwargs["torch_dtype"] = self.torch_dtype
        self.model = self._load_model()
        self.cache_path = tempfile.mkdtemp(prefix="pylate_cache")
        weakref.finalize(self, shutil.rmtree, self.cache_path, ignore_errors=True)
        self.cache = self._create_cache()

    def _load_model(self):
        from pylate.models import ColBERT
        return ColBERT(
            self.reranker_name,
            **self.colbert_kwargs,
            trust_remote_code=True,
            device="cuda",
            model_kwargs=self.model_kwargs
        )

    def _create_cache(self):
        import diskcache as dc
        return dc.Cache(
            self.cache_path,
            size_limit=10*1024**3,
            eviction_policy="least-recently-used",
            sqlite_mmap_size=2**30,
            sqlite_cache_size=2**15
        )

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        queries_cache = self._get_embeddings(queries, is_query=True)
        docs_cache = self._get_embeddings(docs, is_query=False)
        from pylate.scores import colbert_scores_pairwise
        scores = []
        for query, doc in zip(queries, docs):
            query_emb = [queries_cache[query]]
            doc_emb = [docs_cache[doc]]
            score = colbert_scores_pairwise(torch.Tensor(query_emb), torch.Tensor(doc_emb))
            scores.append(score.item())
        return scores

    def _get_embeddings(self, texts: List[str], is_query: bool):
        prefix = "q_" if is_query else "p_"
        cached = {text: self.cache[prefix + text] for text in set(texts) if (prefix + text) in self.cache}
        uncached = list({text for text in texts if text not in cached})
        if len(uncached) == 0:
            return cached
        embeddings = self.model.encode(
            uncached,
            batch_size=self.batch_size,
            is_query=is_query,
            show_progress_bar=False
        )
        for idx, text in enumerate(uncached):
            emb = embeddings[idx]
            self.cache[prefix + text] = emb
            cached[text] = emb
        return cached


class MixedbreadReranker(Reranker):

    def __init__(self, **kwargs):
        install_package("mxbai_rerank", "mxbai-rerank")
        self.reranker_name = kwargs["reranker_name"]
        self.use_bf16 = kwargs.get("bf16", False)
        self.use_fp16 = kwargs.get("fp16", False)
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_seq_length", 8192)
        self.model = self._load_model()

    def _load_model(self):
        from mxbai_rerank import MxbaiRerankV2
        dtype = "auto"
        if self.use_bf16:
            dtype = torch.bfloat16
        elif self.use_fp16:
            dtype = torch.float16
        model = MxbaiRerankV2(self.reranker_name, device="cuda", torch_dtype=dtype, max_length=self.max_length)
        return model

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        scores = self.model._compute_scores(
            queries=queries, documents=docs, batch_size=self.batch_size, show_progress=False
        )
        return scores.tolist()


class RerankerHybrid(HybridStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs
        self.rerank_limit = kwargs.get("rerank_limit", None)
        self.batch_size = kwargs.get("batch_size", 32)
        self.reranker: Optional[Reranker] = None
        self._stats_queries = 0
        self._stats_elapsed = 0.0

    def _load_reranker(self):
        reranker_type = self.args.get("reranker_type", "classifier")
        reranker_name = self.args['reranker_name']
        assert reranker_type in (
            "classifier", "seq2seq", "flag_classifier", "jina", "mixedbread", "qwen3",
            "flag_llm", "flag_layerwise_llm", "flag_lightweight_llm", "pylate"
        )
        logging.info(f"Loading {reranker_type} reranker {reranker_name}")
        if reranker_type == "classifier":
            self.reranker = ClassifierReranker(**self.args)
        elif reranker_type == "seq2seq":
            self.reranker = Seq2SeqReranker(**self.args)
        elif reranker_type in ("flag_classifier", "flag_llm", "flag_layerwise_llm", "flag_lightweight_llm"):
            self.reranker = FlagReranker(**self.args)
        elif reranker_type == "jina":
            self.reranker = JinaReranker(**self.args)
        elif reranker_type == "mixedbread":
            self.reranker = MixedbreadReranker(**self.args)
        elif reranker_type == "qwen3":
            self.reranker = Qwen3Reranker(**self.args)
        elif reranker_type == "pylate":
            self.reranker = PylateReranker(**self.args)

    def merge_results(self, queries: List[IndexInput], index_results: List[Dict], k: int) -> Dict:
        if self.reranker is None:
            self._load_reranker()
        results = {}
        passages = self._load_task_data()
        queries = {query.id: query for query in queries}
        for qid in tqdm(queries.keys(), desc="Reranking results"):
            rerank_func: Callable = self._rerank_limited if self.rerank_limit else self._rerank_all
            results[qid] = rerank_func(qid, queries, passages, index_results, k)
        return results

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        if self.reranker is None:
            self._load_reranker()
        res = []
        for i in range(0, len(docs), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_docs = docs[i:i + self.batch_size]
            pred = self.reranker.rerank_pairs(batch_queries, batch_docs, proba)
            if isinstance(pred, float):
                pred = [pred]
            res.extend(pred)
        return res

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        if self.reranker is None:
            self._load_reranker()
        res = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            pred = self.reranker.rerank(query, batch, proba)
            if isinstance(pred, float):
                pred = [pred]
            res.extend(pred)
        return res

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
        start_time = time.time()
        for i in range(0, len(docids), self.batch_size):
            batch = docids[i:i + self.batch_size]
            docs = [passages[docid].text for docid in batch]
            with torch.no_grad():
                outputs = self.reranker.rerank(query, docs)
                outputs = [outputs] if not isinstance(outputs, list) else outputs
            results.extend([IndexResult(docid, float(score)) for docid, score in zip(batch, outputs)])
        results.sort(key=lambda v: -v.score)
        results = results[:k]
        self._stats_queries += 1
        self._stats_elapsed += (time.time() - start_time)
        return results

    def accumulate_stats(self, stats: Dict):
        q = stats.get("queries", 0)
        t = stats.get("reranking_time", 0.0) + self._stats_elapsed
        stats["reranking_time"] = t
        stats["reranking_qps"] = q / t

    def _load_task_data(self):
        logging.info("Loading passages for reranker")
        return {val.id: val for val in self.task.passages(self.data_dir)}

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
        import xgboost as xgb
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
        import xgboost as xgb
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

    def _validate_model(self, model, grouped: List[Tuple]):
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

    def merge_results(self, queries: List[IndexInput], index_results: List[Dict], k: int) -> Dict:
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
                 strategy: Optional[Union[HybridStrategy, Dict]], task: RetrievalTask, rerank_limit: Optional[int],
                 use_bettertransformer: bool):
        self.data_dir = data_dir
        self.hybrid_name = hybrid_name
        self.indices = indices
        if isinstance(strategy, HybridStrategy):
            self.strategy = strategy
        else:
            self.strategy = self._load_strategy(strategy, rerank_limit, use_bettertransformer)
        self.strategy.task = task
        self.strategy.data_dir = data_dir
        self.k0 = k0

    def _load_strategy(self, spec: Dict, rerank_limit: int, use_bettertransformer: bool):
        spec["rerank_limit"] = rerank_limit
        spec["use_bettertransformer"] = use_bettertransformer
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
        results = self.strategy.merge_results(queries, index_results, k)
        if self.strategy.needs_cache():
            self.save_results_to_cache(k, cache_prefix, results)
        return results

    def results_exist(self, k: int, cache_prefix: str):
        for index in self.indices:
            if not index.results_exist(k, cache_prefix):
                return False
        return True

    def accumulate_stats(self, stats: Dict):
        if isinstance(self.strategy, RerankerHybrid):
            self.strategy.accumulate_stats(stats)

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
