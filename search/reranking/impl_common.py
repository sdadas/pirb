import logging
import os
import shutil
import tempfile
import weakref
from typing import List
import numpy as np
from scipy.special import expit
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel, \
    AutoModelForSeq2SeqLM, AutoModelForCausalLM
from search.base import SmartTemplate
from search.reranking import RerankerBase
from utils.system import install_package


class ClassifierReranker(RerankerBase):

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_name = kwargs["reranker_name"]
        self.fp16 = kwargs.get("fp16", False)
        self.bf16 = kwargs.get("bf16", False)
        self.maxlen = kwargs.get("max_seq_length", 512)
        self.prompt = kwargs.get("prompt", None)
        self.template = kwargs.get("template", "{query}{sep}{sep}{passage}")
        self.use_bettertransformer = kwargs.get("use_bettertransformer", False)
        self.model_kwargs = kwargs.get("model_kwargs", {})
        model, tokenizer = self._load_classifier()
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.sep = self.tokenizer.sep_token
        self.eos = self.tokenizer.eos_token
        assert self.sep is not None or "{sep}" not in self.template, "sep token is none"
        assert self.eos is not None or "{eos}" not in self.template, "eos token is none"

    def _load_classifier(self):
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        dtype = torch.float32
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_name, torch_dtype=dtype, device_map=self.device, **self.model_kwargs
        )
        model.eval()
        if self.use_bettertransformer:
            from opi_optimum.bettertransformer import BetterTransformer
            try:
                model = BetterTransformer.transform(model)
            except NotImplementedError:
                logging.warning(f"Model {model.config.model_type} not supported in BetterTransformer")
        return model, tokenizer

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        assert len(queries) == len(docs)
        texts = [
            self.template.format(query=queries[i], passage=docs[i], sep=self.sep, eos=self.eos, prompt=self.prompt)
            for i in range(len(docs))
        ]
        tokens = self.tokenizer(texts, padding="longest", max_length=self.maxlen, truncation=True, return_tensors="pt")
        tokens.to(self.device)
        output = self.model(**tokens)
        logits = output.logits.detach().type(torch.float32).cpu().numpy()
        logits = np.squeeze(logits)
        return expit(logits).tolist() if proba else logits.tolist()


class VLLMClassifierReranker(RerankerBase):

    def __init__(self, **kwargs):
        self.config = kwargs
        self.reranker_name = kwargs["reranker_name"]
        self.prompt = kwargs.get("prompt", "")
        self.query_template = kwargs.get("query_template")
        self.doc_template = kwargs.get("doc_template")
        self.max_len = self.config["max_seq_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        self.model = self._create_model()

    def _create_model(self):
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM, __version_tuple__
        dtype = "float32"
        if self.config.get("fp16", False):
            dtype = "float16"
        elif self.config.get("bf16", False):
            dtype = "bfloat16"
        extra_args = self.config.get("model_kwargs", {})
        if "max_seq_length" in self.config:
            extra_args["max_model_len"] = self.max_len
        if __version_tuple__ > (0, 10, 0):
            extra_args["runner"] = "pooling"
        else:
            extra_args["task"] = "score"
        args = {
            "model": self.reranker_name,
            "dtype": dtype,
            "enable_chunked_prefill": False,  # chunked prefill broken in VLLM 0.10.1.1, enable it later
            **extra_args
        }
        return LLM(**args)

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        raise NotImplementedError("Pair based reranking not possible for VLLMClassifierReranker")

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        query = self.query_template.format(query=query, prompt=self.prompt)
        docs = [self.doc_template.format(passage=doc, prompt=self.prompt) for doc in docs]
        outputs = self.model.score(query, docs, use_tqdm=False, truncate_prompt_tokens=self.max_len)
        scores = [output.outputs.score for output in outputs]
        return scores


class Seq2SeqReranker(RerankerBase):

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
            from opi_optimum.bettertransformer import BetterTransformer
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


class Qwen3Reranker(RerankerBase):

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


class FlagReranker(RerankerBase):

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


class JinaReranker(RerankerBase):

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


class PylateReranker(RerankerBase):

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


class MixedbreadReranker(RerankerBase):

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


class CtxlReranker(RerankerBase):

    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.use_bf16 = kwargs.get("bf16", False)
        self.use_fp16 = kwargs.get("fp16", False)
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_seq_length", 8192)
        self.model_kwargs = kwargs.get("model_kwargs", {})
        model, tokenizer = self._load_model()
        self.model = model
        self.tokenizer = tokenizer

    def _load_model(self):
        dtype = "auto"
        if self.use_bf16:
            dtype = torch.bfloat16
        elif self.use_fp16:
            dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.reranker_name, device_map="cuda", torch_dtype=dtype, **self.model_kwargs
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        prompts = self._format_prompts(queries, docs)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            next_logits = out.logits[:, -1, :]
            scores = next_logits[:, 0].detach().float().cpu().tolist()
            return scores

    def _format_prompts(self, queries: List[str], docs: List[str], instruction: str = ""):
        prompts = []
        if instruction:
            instruction = f" {instruction}"
        for query, doc in zip(queries, docs):
            prompt = (f"Check whether a given document contains information helpful to answer "
                      f"the query.\n<Document> {doc}\n<Query> {query}{instruction} ??")
            prompts.append(prompt)
        return prompts


class CrossEncoderReranker(RerankerBase):
    def __init__(self, **kwargs):
        self.reranker_name = kwargs["reranker_name"]
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_seq_length", 8192)
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.trust_remote_code = self.model_kwargs.get("trust_remote_code", True)
        self.revision = self.model_kwargs.get("revision", None)
        self.model = self._load_model()

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(self.reranker_name, device="cuda",
                             trust_remote_code=self.trust_remote_code,
                             max_length=self.max_length,
                             revision=self.revision)
        return model

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        pairs = list(zip(queries, docs))
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return scores.tolist()

