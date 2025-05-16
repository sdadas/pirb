import json
import json
import logging
import os.path
import shutil
from typing import List, Iterable, Dict, Optional, TextIO
import numpy as np
import torch
from jnius import autoclass
from pyserini.encode import QueryEncoder
from pyserini.search import LuceneImpactSearcher
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, BatchEncoding
from data import IndexInput, IndexResult
from search.base import SearchIndex


class SpladeEncoder(QueryEncoder):

    def __init__(self, config: Dict, use_bettertransformer: bool, quantization_factor: int = 100):
        self.quantization_factor = quantization_factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.use_bettertransformer = use_bettertransformer
        self.model_name_or_path = config["name"]
        self.maxlen = config.get("max_seq_length", 512)
        self.tokenizer, self.model = self._init_tokenizer_and_model()
        self.vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.queries_cache = {}

    def _init_tokenizer_and_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name_or_path).to(self.device)
        tokenizer.model_max_length = self.maxlen
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        model.eval()
        if self.config.get("fp16", False):
            model.half()
        return tokenizer, model

    def encode_batch(self, batch: List[str]):
        batch: BatchEncoding = self.tokenizer(
            batch, padding="longest", truncation=True, add_special_tokens=True,
            return_tensors="pt", max_length=self.tokenizer.model_max_length
        ).to(self.device)
        output = self.model(**batch)
        logits, attention_mask = output["logits"].detach(), batch["attention_mask"].detach()
        attention_mask = attention_mask.unsqueeze(-1)
        _relu = torch.relu(logits)
        splade_vectors = torch.max(torch.log(torch.add(_relu, 1)) * attention_mask, dim=1)
        input_ids = batch["input_ids"]
        input_ids.detach()
        del batch, logits, attention_mask, input_ids, _relu
        result = splade_vectors[0].detach().squeeze()
        splade_vectors[1].detach()
        del splade_vectors
        return result

    def encode_document_vector(self, splade_vector: torch.Tensor):
        data = splade_vector.cpu().numpy()
        idx = np.nonzero(data)
        data = data[idx]
        data = np.rint(data * self.quantization_factor).astype(int)
        dict_splade = dict()
        for id_token, value_token in zip(idx[0], data):
            if value_token > 0:
                real_token = self.vocab[id_token]
                dict_splade[real_token] = int(value_token)
        return dict_splade

    def encode_queries(self, queries: List[str], batch_size: int):
        cached = {}
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            encoded = self.encode_batch(batch).cpu().numpy()
            outputs = self._output_to_weight_dicts(encoded)
            assert len(batch) == len(outputs)
            cached.update({text: output for text, output in zip(batch, outputs)})
        self.queries_cache = cached

    def _output_to_weight_dicts(self, batch_aggregated_logits: torch.Tensor, max_clauses: int = 1024):
        to_return = []
        if len(batch_aggregated_logits.shape) == 1:
            batch_aggregated_logits = [batch_aggregated_logits]
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            outputs = [(self.vocab[k], float(v)) for k, v in zip(list(col), list(weights))]
            outputs = sorted(outputs, key=lambda v: -v[1])[:max_clauses]
            d = {k: v for k, v in outputs}
            to_return.append(d)
        return to_return

    def encode(self, text, max_length=512, **kwargs):
        return self.queries_cache[text]


class SpladeIndex(SearchIndex):

    def __init__(self, config: Dict, data_dir: str, threads: int, use_bettertransformer: bool = False):
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.data_dir = data_dir
        self.model_name_or_path = config["name"]
        self.index_name = self.model_name_or_path.replace("/", "_").replace(".", "_")
        self.base_dir = os.path.join(self.data_dir, self.index_name)
        self.index_dir = os.path.join(self.base_dir, 'lucene_index')
        self.use_bettertransformer = use_bettertransformer
        self.threads = threads
        self.encoder: Optional[SpladeEncoder] = None
        self.searcher = None

    def _init_encoder(self):
        self.encoder = SpladeEncoder(self.config, self.use_bettertransformer)

    def exists(self) -> bool:
        return os.path.exists(self.index_dir)

    def build(self, docs: Iterable[IndexInput]):
        if self.exists(): shutil.rmtree(self.index_dir)
        if self.encoder is None: self._init_encoder()
        docs_dir = os.path.join(self.base_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        logging.info("Building splade vectors index %s", self.index_dir)
        with open(os.path.join(docs_dir, "passages.jsonl"), "w", encoding='utf-8') as docs_file:
            with torch.no_grad():
                batch = []
                for doc in docs:
                    batch.append(doc)
                    if len(batch) >= self.batch_size:
                        self._write_batch(batch, docs_file)
                        batch = []
                if len(batch) > 0:
                    self._write_batch(batch, docs_file)
        os.makedirs(self.index_dir, exist_ok=True)
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

    def _write_batch(self, batch: List[IndexInput], output: TextIO):
        texts = [val.text for val in batch]
        splade_batch = self.encoder.encode_batch(texts)
        if len(texts) == 1:
            splade_batch = splade_batch.unsqueeze(0)
        for doc, splade_vector in zip(batch, splade_batch):
            splade_dict = self.encoder.encode_document_vector(splade_vector)
            dict_ = dict(id=doc.id, content="", vector=splade_dict)
            json_dict = json.dumps(dict_, ensure_ascii=False)
            output.write(json_dict + "\n")

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None, overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self.searcher is None: self._open_searcher()
        results = {}
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching in splade index", disable=not verbose):
            batch: List[IndexInput] = queries[i:i + batch_size]
            texts = [val.text for val in batch]
            ids = [val.id for val in batch]
            with torch.no_grad():
                self.encoder.encode_queries(texts, batch_size=self.batch_size)
            batch_results = self.searcher.batch_search(texts, ids, k=k, threads=self.threads)
            for qid, relevant in batch_results.items():
                batch_results[qid] = [IndexResult(val.docid, val.score) for val in relevant]
            results.update(batch_results)
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def _open_searcher(self):
        if self.encoder is None: self._init_encoder()
        searcher = LuceneImpactSearcher(self.index_dir, self.encoder)
        self.searcher = searcher

    def name(self):
        return f"splade_{self.index_name}"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return self.config

    def index_path(self) -> str:
        return self.index_dir


