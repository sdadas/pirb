import logging
import re
from collections.abc import Iterable
from copy import deepcopy
from itertools import batched
from pathlib import Path

import torch
from scipy.sparse import csr_array
from sentence_transformers import SparseEncoder
from tqdm import tqdm

from backend import IndexBackend
from data import IndexInput, IndexResult
from search.base import SearchIndex


class SparseIndex(SearchIndex):

    def __init__(self, data_dir: str, encoder: dict):
        self.data_dir = data_dir
        self.encoder_spec = deepcopy(encoder)
        self.encoder = None
        model_name = encoder["name"].replace("\\", "/").rstrip("/")
        slug = re.sub(r"[^\w-]+", "_", model_name.rsplit("/", 1)[-1])
        self.index_name = f"sparse_{slug[:64]}"
        self.index_dir = str(Path(data_dir) / self.index_name)

        backend = encoder.get("backend", "sparse_csr")
        backend_config = {"backend_type": backend} if isinstance(backend, str) else dict(backend)
        backend_type = backend_config["backend_type"]
        if backend_type not in ("sparse_csr", "seismic"):
            raise ValueError(f"Unsupported SparseIndex backend: {backend_type}")
        self.backend = IndexBackend.from_config(dict(backend_config, index_dir=self.index_dir))

    def _load_encoder(self):
        if self.encoder is None:
            spec = self.encoder_spec
            model_kwargs = dict(spec.get("model_kwargs") or {})
            trust_remote_code = model_kwargs.pop("trust_remote_code", False)
            model = SparseEncoder(
                spec["name"],
                device=spec.get("device"),
                revision=spec.get("revision"),
                trust_remote_code=spec.get("trust_remote_code", trust_remote_code),
                model_kwargs=model_kwargs,
                config_kwargs=spec.get("config_kwargs"),
            )
            if model.similarity_fn_name != "dot":
                raise ValueError(f"SparseIndex requires dot similarity, got {model.similarity_fn_name!r}")
            self.encoder = model
        return self.encoder

    def _encode_options(self, side):
        spec = self.encoder_spec
        options = {
            "batch_size": spec.get("batch_size", 32),
            "show_progress_bar": False,
            **spec.get("encode_kwargs", {}),
            **spec.get(f"{side}_encode_kwargs", {}),
        }
        prefix = spec.get(f"{side}_prefix")
        prefix_name = spec.get(f"{side}_prefix_name")
        if prefix is not None or prefix_name is not None:
            options.pop("prompt", None)
            options.pop("prompt_name", None)
            options.update(prompt=prefix, prompt_name=prefix_name if prefix is None else None)
        options.update(convert_to_sparse_tensor=True, convert_to_tensor=True, save_to_cpu=True)
        return options

    @staticmethod
    def _encode_batch(encode, batch, options):
        ids = [item.id for item in batch]
        previous_threads = torch.get_num_threads()
        # Splitting sparse CPU tensors into rows is slower with large thread pools.
        limit_threads = previous_threads > 1
        if limit_threads:
            torch.set_num_threads(1)
        try:
            embeddings = encode([item.text for item in batch], **options).coalesce()
            rows, columns = embeddings.indices().numpy()
            values = embeddings.values().float().numpy()
        finally:
            if limit_threads:
                torch.set_num_threads(previous_threads)
        return csr_array((values, (rows, columns)), shape=embeddings.shape), ids

    def build(self, docs: Iterable[IndexInput]):
        model = self._load_encoder()
        options = self._encode_options("p")
        logging.info("Building sparse index %s", self.index_dir)
        self.backend.build(
            self._encode_batch(model.encode_document, batch, options)
            for batch in batched(docs, options["batch_size"] * 16)
        )

    def search(self, queries: list[IndexInput], k: int, batch_size=1024, verbose=True,
               cache_prefix=None, overwrite=False) -> dict[str, list[IndexResult]]:
        if not overwrite:
            results = self.load_results_from_cache(k, cache_prefix)
            if results is not None:
                return results

        results = {}
        if queries and k > 0:
            model = self._load_encoder()
            options = self._encode_options("q")
            for start in tqdm(range(0, len(queries), batch_size), desc="Searching in sparse index", disable=not verbose):
                embeddings, ids = self._encode_batch(model.encode_query, queries[start:start + batch_size], options)
                results.update(self.backend.search(embeddings, ids, k))
        elif queries:
            results = {query.id: [] for query in queries}
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def exists(self):
        return self.backend.exists()

    def name(self):
        return self.index_name

    def model_dict(self):
        return self.encoder_spec

    def index_path(self):
        return self.index_dir
