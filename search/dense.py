import logging
import math
import os.path
from collections import defaultdict
from typing import List, Iterable, Dict, Optional, Union
import faiss
import numpy as np
import torch
from joblib import load
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data import IndexInput, IndexResult
from search.base import SearchIndex, patch_sentence_transformer


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
                    doc_res = doc_res / norm
            res.append(doc_res)
        return torch.vstack(res) if convert_to_tensor else np.vstack(res)


class DenseIndex(SearchIndex):

    def __init__(self, data_dir: str, encoder: Dict, normalize=True, use_bettertransformer=False, raw_mode=True):
        self._st_bs = encoder.get("batch_size", 32)
        self._averaging = encoder.get("enable_averaging", False)
        self.data_dir = data_dir
        self.index_name = encoder["name"].replace("/", "_").replace(".", "_")
        if self._averaging:
            self.index_name += "_averaging"
        self.index_dir = os.path.join(self.data_dir, self.index_name)
        self.index_vectors_path = os.path.join(self.index_dir, "index.bin")
        self.index_ids_path = os.path.join(self.index_dir, "ids.bin")
        self.normalize = normalize
        self.raw_mode = raw_mode
        self.use_bettertransformer = use_bettertransformer
        self.encoder_spec = encoder
        self.query_prefix: str = encoder.get("q_prefix", None)
        self.passage_prefix: str = encoder.get("p_prefix", None)
        self.encoder: Optional[SentenceTransformer] = None
        self._ids = None
        self._index = None

    def _load_encoder(self):
        if self.encoder is not None: return
        torch_dtype = torch.float32
        if self.encoder_spec.get("fp16", False):
            torch_dtype = torch.float16
        elif self.encoder_spec.get("bf16", False):
            torch_dtype = torch.bfloat16
        model = SentenceTransformer(self.encoder_spec["name"], model_kwargs={"torch_dtype": torch_dtype})
        if self.encoder_spec.get("fp16", False):
            model.half()
        elif self.encoder_spec.get("bf16", False):
            model.bfloat16()
        if "max_seq_length" in self.encoder_spec:
            model.max_seq_length = int(self.encoder_spec["max_seq_length"])
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

    def exists(self) -> bool:
        return os.path.exists(self.index_vectors_path) and os.path.exists(self.index_ids_path)

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
            batch_emb = self.encoder.encode(
                batch, normalize_embeddings=self.normalize, convert_to_tensor=self.raw_mode,
                batch_size=self._st_bs, prompt=self.passage_prefix
            )
            idx += 1
            if self.raw_mode:
                embeddings.append(batch_emb.detach().cpu())
            else:
                embeddings.append(batch_emb)
        res = self.save_index(self.index_ids_path, self.index_vectors_path, ids, embeddings, self.raw_mode)
        self._index = res
        self._ids = ids

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None,
               overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        if self._ids is None or self._index is None: self._open_index()
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
        emb = self.encoder.encode(
            texts, show_progress_bar=False, normalize_embeddings=self.normalize,
            batch_size=self._st_bs, convert_to_tensor=self.raw_mode, prompt=self.query_prefix
        )
        return self._search_in_memory(emb, qids, top_k) if self.raw_mode else self._search_in_faiss(emb, qids, top_k)

    def _search_in_memory(self, emb, qids, top_k):
        emb = emb.detach().cpu().float()
        results = defaultdict(list)
        hits = semantic_search(emb, self._index, query_chunk_size=128, top_k=top_k)
        for i, qid in enumerate(qids):
            query_hits = hits[i]
            for k in range(top_k):
                hit = query_hits[k]
                docid = self._ids[int(hit["corpus_id"])]
                score = hit["score"]
                results[qid].append(IndexResult(docid, score))
        return results

    def _search_in_faiss(self, emb, qids, top_k):
        results = defaultdict(list)
        sim, indices = self._index.search(emb, top_k)
        sim = sim.tolist()
        indices = indices.tolist()
        for i, qid in enumerate(qids):
            for k in range(top_k):
                idx = indices[i][k]
                docid = self._ids[idx]
                score = sim[i][k]
                results[qid].append(IndexResult(docid, score))
        return results

    def _open_index(self):
        if self.raw_mode:
            self._index = torch.load(self.index_vectors_path).float()
        else:
            self._index = faiss.read_index(self.index_vectors_path)
        self._ids = load(self.index_ids_path)

    def name(self):
        return f"dense_{self.index_name}"

    def __len__(self):
        return len(self._ids) if self._ids is not None else 0

    def model_dict(self) -> Dict:
        return self.encoder_spec
