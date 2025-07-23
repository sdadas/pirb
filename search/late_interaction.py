import logging
import os
from typing import Dict, Optional, List, Iterable

import torch
from sentence_transformers import SentenceTransformer

from backend import IndexBackend
from data import IndexInput, IndexResult
from search import SearchIndex

class ColBertEmbeddings:

    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.model_name = self.config["name"]
        self.document_length = config.get("document_length", None)
        self.query_length = config.get("query_length", None)
        self.query_prefix: Optional[str] = config.get("q_prefix", None)
        self.passage_prefix: Optional[str] = config.get("p_prefix", None)
        self.model_kwargs = config.get("model_kwargs", {})
        self.model = self._create_model()

    def _create_model(self):
        from pylate import models
        torch_dtype = torch.float32
        if self.config.get("fp16", False):
            torch_dtype = torch.float16
        elif self.config.get("bf16", False):
            torch_dtype = torch.bfloat16
        model_kwargs = {"torch_dtype": torch_dtype}
        model_kwargs.update(self.model_kwargs)
        extra_args = self.config.get("model_kwargs", {})
        args = {
            "model_name_or_path": self.model_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "query_prefix": self.query_prefix,
            "document_prefix": self.passage_prefix,
            **extra_args
        }
        model = models.ColBERT(**args, model_kwargs=self.model_kwargs)
        if self.document_length is not None:
            model.document_length = self.document_length
        if self.query_length is not None:
            model.query_length = self.query_length
        model.eval()
        return model

    def encode(self, docs: List[str], is_query: bool):
        return self.model.encode(
            sentences=docs,
            batch_size=self.batch_size,
            is_query=is_query,
            show_progress_bar=True
        )


class LateInteractionIndex(SearchIndex):

    def __init__(self, data_dir: str, encoder: Dict):
        self.data_dir = data_dir
        self.index_name = encoder["name"].replace("/", "_").replace(".", "_")
        self.model_kwargs = encoder.get("model_kwargs", {})
        self.index_dir = os.path.join(self.data_dir, self.index_name)
        self.backend = self._create_backend(encoder)
        self.encoder_spec = encoder
        self.query_prefix: Optional[str] = encoder.get("q_prefix", None)
        self.passage_prefix: Optional[str] = encoder.get("p_prefix", None)
        self.encoder: Optional[ColBertEmbeddings] = None

    def _create_backend(self, encoder_spec: Dict):
        backend = encoder_spec.get("backend", "plaid")
        if isinstance(backend, str):
            return IndexBackend.from_config({"backend_type": backend, "index_dir": self.index_dir})
        elif isinstance(backend, dict):
            backend["index_dir"] = self.index_dir
            return IndexBackend.from_config(backend)
        raise AssertionError("'backend' arg accepts backend name or backend config as dict")

    def exists(self) -> bool:
        return self.backend.exists()

    def _load_encoder(self):
        if self.encoder is None:
            self.encoder = ColBertEmbeddings(self.encoder_spec)
        return self.encoder

    def build(self, docs: Iterable[IndexInput]):
        if self.encoder is None:
            self.encoder = self._load_encoder()
        ids, texts = [], []
        for doc in docs:
            ids.append(doc.id)
            texts.append(doc.text)
        logging.info("Building late_interaction index %s", self.index_dir)
        embeddings = self.encoder.encode(texts, is_query=False)
        self.backend.build(embeddings, ids)

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None,
               overwrite=False) -> Dict[str, List[IndexResult]]:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self.encoder is None: self._load_encoder()
        texts = [val.text for val in queries]
        qids = [val.id for val in queries]
        emb_q = self.encoder.encode(texts, is_query=True)
        return self.backend.search(emb_q, qids, k)

    def name(self):
        return f"late_interaction_{self.index_name}"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return self.encoder_spec

    def index_path(self) -> str:
        return self.index_dir

    def __del__(self):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.close()
