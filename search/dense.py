import logging
import math
import os.path
import random
from functools import partial
from typing import List, Iterable, Dict, Optional, Union, Any, Callable
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from multiprocessing import Pool
from backend import IndexBackend
from data import IndexInput
from search.base import SearchIndex, patch_sentence_transformer


class VLLMEmbeddings:

    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.max_len = self.config["max_seq_length"] if "max_seq_length" in self.config else None
        self.model_name = self.config["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._create_model()

    def _create_model(self):
        from vllm import LLM
        dtype = "float32"
        if self.config.get("fp16", False):
            dtype = "float16"
        elif self.config.get("bf16", False):
            dtype = "bfloat16"
        extra_args = self.config.get("model_kwargs", {})
        if "max_seq_length" in self.config:
            extra_args["max_model_len"] = self.config["max_seq_length"]
        args = {
            "model": self.model_name,
            "dtype": dtype,
            "task": "embed",
            "trust_remote_code": True,
            **extra_args
        }
        return LLM(**args)

    def encode(self, batch: Union[str, List[str]], convert_to_tensor: bool = False, **kwargs):
        if isinstance(batch, str):
            batch = [batch]
        if kwargs.get("prompt", None) is not None:
            prompt = kwargs["prompt"]
            batch = [prompt + val for val in batch]
        pbar = tqdm(total=len(batch), disable=not kwargs.get("show_progress_bar", True))
        results = []
        for i in range(0, len(batch), self.batch_size):
            mini_batch = batch[i:i + self.batch_size]
            if self.max_len is not None:
                mini_batch = self._truncate_prompt_tokens(mini_batch)
            outputs = self.model.embed(mini_batch, use_tqdm=False)
            result = [val.outputs.embedding for val in outputs]
            if convert_to_tensor:
                embeddings = torch.tensor(result, dtype=torch.float32)
            else:
                embeddings = np.array(result, dtype=np.float32)
            results.append(embeddings)
            pbar.update(self.batch_size)
        pbar.close()
        return torch.vstack(results) if convert_to_tensor else np.vstack(results)

    def _truncate_prompt_tokens(self, batch):
        truncated = []
        for i in range(0, len(batch)):
            tokens = self.tokenizer.encode(batch[i], add_special_tokens=True, truncation=True, max_length=self.max_len)
            texts = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            truncated.append(texts)
        return truncated


class OpenAIEmbeddings:

    def __init__(self, config: Dict):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.config = config
        self.model = config["name"]
        self.api_name = config.get("api_name", self.model)
        self.api_base = config["api_base"]
        self.api_key = config.get("api_key", "-")
        self.batch_size = config.get("batch_size", 32)
        self.tokenizer: Any = self._create_tokenizer(self.model)
        self.truncate_func: Callable = (
            self._truncate_prompt_tokens_hf
            if isinstance(self.tokenizer, PreTrainedTokenizer)
            else self._truncate_prompt_tokens_tiktoken
        )
        self.max_len = config.get("model_max_length")
        self.threads = config.get("threads", 8)
        logging.info("Using remote embeddings via OpenAI API")

    def _create_tokenizer(self, model_name: str):
        if model_name.startswith("openai/"):
            import tiktoken
            model_name = model_name.removeprefix("openai/")
            return tiktoken.encoding_for_model(model_name)
        else:
            return AutoTokenizer.from_pretrained(model_name)

    def encode(self, batch: Union[str, List[str]], convert_to_tensor: bool = False, **kwargs):
        if isinstance(batch, str):
            batch = [batch]
        if kwargs.get("prompt", None) is not None:
            prompt = kwargs["prompt"]
            batch = [prompt + val for val in batch]
        results = []
        pbar = tqdm(total=len(batch), disable=not kwargs.get("show_progress_bar", True))
        with Pool(processes=self.threads) as executor:
            generator = list(self._get_chunks(batch))
            partial_func = partial(
                OpenAIEmbeddings._encode_batch, api_name=self.api_name, api_base=self.api_base, api_key=self.api_key
            )
            for thread_result in executor.imap(partial_func, generator):
                if convert_to_tensor:
                    embeddings = torch.tensor(thread_result, dtype=torch.float32)
                else:
                    embeddings = np.array(thread_result, dtype=np.float32)
                results.append(embeddings)
                pbar.update(self.batch_size)
        pbar.close()
        return torch.vstack(results) if convert_to_tensor else np.vstack(results)

    def _get_chunks(self, data: List[str]):
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch = [(val if len(val.strip()) > 0 else "?") for val in batch]
            yield self.truncate_func(batch, self.max_len)

    def _truncate_prompt_tokens_hf(self, batch, max_len):
        truncated = []
        for i in range(0, len(batch)):
            tokens = self.tokenizer.encode(batch[i], add_special_tokens=True, truncation=True, max_length=max_len)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            truncated.append(text)
        return truncated

    def _truncate_prompt_tokens_tiktoken(self, batch, max_len):
        truncated = []
        for i in range(0, len(batch)):
            tokens = self.tokenizer.encode(batch[i])
            tokens = tokens[:max_len]
            text = self.tokenizer.decode(tokens)
            truncated.append(text)
        return truncated

    @staticmethod
    def _encode_batch(batch: List[str], api_name: str, api_base: Union[List, str], api_key: str):
        from openai import OpenAI
        if isinstance(api_base, list):
            api_base = random.choice(api_base)
        client = OpenAI(api_key=api_key, base_url=api_base)
        result = client.embeddings.create(
            input=batch, model=api_name, encoding_format="float", timeout=300
        )
        embeddings = [val.embedding for val in result.data]
        return embeddings


class TextAveragingEncoder:

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.tokenizer: PreTrainedTokenizer = self.encoder.tokenizer
        self.max_len = self.encoder.max_seq_length
        logging.info("Using text averaging encoder")

    def encode(self, batch: Union[str, List[str]], convert_to_tensor: bool = False, **kwargs):
        if isinstance(batch, str):
            batch = [batch]
        chunked_inputs, doc_pos = self._chunk_inputs(batch)
        embeddings = self.encoder.encode(chunked_inputs, convert_to_tensor=convert_to_tensor, **kwargs)
        return self._build_embeddings(embeddings, len(chunked_inputs), doc_pos, convert_to_tensor)

    def _chunk_inputs(self, batch: List[str]):
        encodings = self.tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        chunked_inputs = []
        doc_pos = []
        for encoding in encodings.encodings:
            seq = encoding.ids
            doc_pos.append(len(chunked_inputs))
            for i in range(0, len(seq), self.max_len):
                chunk_seq = seq[i:i + self.max_len]
                chunked_inputs.append(self.tokenizer.decode(chunk_seq))
        return chunked_inputs, doc_pos

    def _build_embeddings(self, embeddings, size: int, doc_pos: List[int], convert_to_tensor: bool):
        res = []
        for idx in range(len(doc_pos)):
            start_pos = doc_pos[idx]
            end_pos = doc_pos[idx + 1] if idx < len(doc_pos) - 1 else size
            doc_emb = embeddings[start_pos:end_pos, :]
            if convert_to_tensor:
                doc_emb = doc_emb.detach().cpu()
                doc_res = torch.mean(doc_emb, dim=0, dtype=torch.float32)
                doc_res = torch.nn.functional.normalize(doc_res, p=2, dim=0)
            else:
                doc_res = np.mean(doc_emb, axis=0)
                norm = np.linalg.norm(doc_res)
                if norm != 0:
                    # noinspection PyTypeChecker
                    doc_res = doc_res / norm
            res.append(doc_res)
        return torch.vstack(res) if convert_to_tensor else np.vstack(res)


class DenseIndex(SearchIndex):

    def __init__(self, data_dir: str, encoder: Dict, use_bettertransformer=False):
        self._st_bs = encoder.get("batch_size", 32)
        self._averaging = encoder.get("enable_averaging", False)
        self.data_dir = data_dir
        self.index_name = encoder["name"].replace("/", "_").replace(".", "_")
        self.model_kwargs = encoder.get("model_kwargs", {})
        self.padding_side = encoder.get("padding_side", None)
        if self._averaging:
            self.index_name += "_averaging"
        self.index_dir = os.path.join(self.data_dir, self.index_name)
        self.backend = self._create_backend(encoder)
        self.normalize = encoder.get("normalize", True)
        self.use_bettertransformer = use_bettertransformer
        self.encoder_spec = encoder
        self.query_prefix: Optional[str] = encoder.get("q_prefix", None)
        self.passage_prefix: Optional[str] = encoder.get("p_prefix", None)
        self.query_prefix_name: Optional[str] = encoder.get("q_prefix_name", None)
        self.passage_prefix_name: Optional[str] = encoder.get("p_prefix_name", None)
        self.encoder: Optional[SentenceTransformer] = None

    def _create_backend(self, encoder_spec: Dict):
        backend = encoder_spec.get("backend", "raw")
        if isinstance(backend, str):
            return IndexBackend.from_config({"backend_type": backend, "index_dir": self.index_dir})
        elif isinstance(backend, dict):
            backend["index_dir"] = self.index_dir
            return IndexBackend.from_config(backend)
        raise AssertionError("'backend' arg accepts backend name or backend config as dict")

    def _load_encoder(self):
        if self.encoder is not None: return
        if self.encoder_spec.get("type", None) == "api":
            self.encoder = OpenAIEmbeddings(self.encoder_spec)
            return
        if self.encoder_spec.get("type", None) == "vllm":
            self.encoder = VLLMEmbeddings(self.encoder_spec)
            return
        torch_dtype = torch.float32
        if self.encoder_spec.get("fp16", False):
            torch_dtype = torch.float16
        elif self.encoder_spec.get("bf16", False):
            torch_dtype = torch.bfloat16
        model_kwargs = {"torch_dtype": torch_dtype}
        model_kwargs.update(self.model_kwargs)
        trust = model_kwargs.get("trust_remote_code", True)
        # noinspection PyArgumentList
        model = SentenceTransformer(self.encoder_spec["name"], trust_remote_code=trust, model_kwargs=model_kwargs)
        if self.encoder_spec.get("fp16", False):
            model.half()
        elif self.encoder_spec.get("bf16", False):
            model.bfloat16()
        if "max_seq_length" in self.encoder_spec:
            model.max_seq_length = int(self.encoder_spec["max_seq_length"])
        if self.padding_side is not None:
            model.tokenizer.padding_side = self.padding_side
        if self.use_bettertransformer:
            patched = patch_sentence_transformer(model)
            if patched:
                logging.info("Using encoder with BetterTransformer enabled")
                if "batch_size" not in self.encoder_spec:
                    self._st_bs = 256
        model.eval()
        if self._averaging:
            model = TextAveragingEncoder(model)
        self.encoder = model

    def _encode_kwargs(self, query: bool) -> Dict:
        kwargs = {
            "normalize_embeddings": self.normalize,
            "batch_size": self._st_bs,
            "convert_to_tensor": self.backend.supports_tensors()
        }
        prefix = self.query_prefix if query else self.passage_prefix
        prefix_name = self.query_prefix_name if query else self.passage_prefix_name
        if prefix:
            kwargs["prompt"] = prefix
        elif prefix_name:
            kwargs["prompt_name"] = prefix_name
            if "jina-embeddings-v3" in self.index_name:
                kwargs["task"] = prefix_name
        causal_fix = "gte-qwen2" in self.index_name.lower() or "inf-retriever" in self.index_name.lower()
        if causal_fix and not isinstance(self.encoder, OpenAIEmbeddings):
            kwargs["is_causal"] = False
            self.encoder.module_kwargs["0"] = ["is_causal"]
        return kwargs

    def exists(self) -> bool:
        return self.backend.exists()

    def build(self, docs: Iterable[IndexInput]):
        if self.encoder is None:
            self._load_encoder()
        ids, texts = [], []
        for doc in docs:
            ids.append(doc.id)
            texts.append(doc.text)
        logging.info("Building dense index %s", self.index_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        embeddings = []
        mega_batch_size = 1_048_576  # split large datasets into parts to avoid cuda OOM
        parts_num = int(math.ceil(len(texts) / mega_batch_size))
        idx = 1
        for i in range(0, len(texts), mega_batch_size):
            if parts_num > 1:
                logging.info(f"Encoding data part {idx} of {parts_num}")
            batch = texts[i:i + mega_batch_size]
            kwargs = self._encode_kwargs(query=False)
            batch_emb = self.encoder.encode(batch, **kwargs)
            idx += 1
            if self.backend.supports_tensors():
                embeddings.append(batch_emb.detach().cpu())
            else:
                embeddings.append(batch_emb)
        self.backend.build(embeddings, ids)

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None,
               overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self.encoder is None: self._load_encoder()
        results = {}
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching in dense index", disable=not verbose):
            batch = queries[i:i + batch_size]
            batch_results = self._search_vectors(batch, k)
            results.update(batch_results)
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def _search_vectors(self, batch: List[IndexInput], top_k: int) -> Dict:
        texts = [val.text for val in batch]
        qids = [val.id for val in batch]
        kwargs = self._encode_kwargs(query=True)
        emb = self.encoder.encode(texts, show_progress_bar=False, **kwargs)
        return self.backend.search(emb, qids, top_k)

    def name(self):
        return f"dense_{self.index_name}"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return self.encoder_spec

    def index_path(self) -> str:
        return self.index_dir

    def __del__(self):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.close()
