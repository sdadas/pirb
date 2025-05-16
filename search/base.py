import gzip
import itertools
import json
import os.path
import re
from abc import ABC
from typing import List, Iterable, Dict, Optional, Tuple
import faiss
import numpy as np
import torch
from joblib import dump
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer

from data import IndexInput, IndexResult


def patch_sentence_transformer(model: SentenceTransformer):
    try:
        from optimum.bettertransformer import BetterTransformer
        from sentence_transformers.models import Transformer
        for module in model.modules():
            if isinstance(module, Transformer):
                module.auto_model = BetterTransformer.transform(module.auto_model)
                return True
    except (ImportError, Exception):
        pass
    return False


class SearchIndex(ABC):

    def exists(self) -> bool:
        raise NotImplementedError()

    def build(self, docs: Iterable[IndexInput]):
        raise NotImplementedError()

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None,
               overwrite=False) -> Dict[str, List[IndexResult]]:
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()

    def basedir(self):
        return getattr(self, "data_dir")

    def results_cache(self, k: int, cache_prefix: str):
        return f"{cache_prefix}_k{k}_{self.name()}.jsonl.gz"

    def results_exist(self, k: int, cache_prefix: str):
        cache_dir = os.path.join(self.basedir(), "results")
        output_file = os.path.join(cache_dir, self.results_cache(k, cache_prefix))
        return os.path.exists(output_file)

    def load_results_from_cache(self, k: int, cache_prefix: str) -> Optional[Dict]:
        if not cache_prefix:
            return None
        cache_dir = os.path.join(self.basedir(), "results")
        output_file = os.path.join(cache_dir, self.results_cache(k, cache_prefix))
        if os.path.exists(output_file):
            results = {}
            with gzip.open(output_file, "rt", encoding="utf-8") as input_file:
                for line in input_file:
                    result = json.loads(line.strip())
                    query_id = result["id"]
                    num_hits = len(result["hits"])
                    hits = [IndexResult(result["hits"][i], result["hits_scores"][i]) for i in range(num_hits)]
                    results[query_id] = hits
            return results
        return None

    def save_results_to_cache(self, k: int, cache_prefix: str, results: Dict):
        if not cache_prefix:
            return
        cache_dir = os.path.join(self.basedir(), "results")
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, self.results_cache(k, cache_prefix))
        tmp_path = output_path + "_tmp"
        with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=3) as output_file:
            for key, val in results.items():
                result = {"id": key, "hits": [v.id for v in val], "hits_scores": [v.score for v in val]}
                output_file.write(json.dumps(result, ensure_ascii=False))
                output_file.write("\n")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp_path, output_path)

    def save_index(self, index_ids_path, index_vectors_path, ids, embeddings, raw_mode):
        if raw_mode:
            res = torch.vstack(embeddings)
            torch.save(res, index_vectors_path)
            dump(ids, index_ids_path)
            res = res.float()
        else:
            embeddings = np.vstack(embeddings)
            dim = embeddings.shape[1]
            res = faiss.IndexFlatIP(dim)
            res.add(embeddings)
            faiss.write_index(res, index_vectors_path)
            dump(ids, index_ids_path)
        return res

    def accumulate_stats(self, stats: Dict):
        pass

    def model_dict(self) -> Dict:
        raise NotImplementedError()

    def index_path(self) -> str:
        raise None


class SmartTemplate:

    def __init__(self, template: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.template = template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_ids, self.placeholders, self.pad_positions = self._prepare_template()

    def _prepare_template(self) -> Tuple[List[int], Dict[int, str], List[int]]:
        pad, pad_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        placeholders = {}

        def replace_placeholder(m):
            value = m.group(0)
            placeholders[len(placeholders)] = value.strip("{}")
            return pad
        replaced = re.sub(r"\{\w+\}", replace_placeholder, self.template)
        template_ids = self.tokenizer.encode(replaced, padding=False, truncation=False)
        pad_positions = [idx for idx, val in enumerate(template_ids) if val == pad_id]
        assert len(pad_positions) == len(placeholders)
        return template_ids, placeholders, pad_positions

    def encode(self, **kwargs):
        encoded = {
            key: self.tokenizer.encode(val, add_special_tokens=False, padding=False, truncation=False)
            for key, val in kwargs.items()
        }
        return self.format_encoded(**encoded)

    def format_encoded(self, **kwargs):
        truncated = self._truncate_args(**kwargs)
        parts = []
        current_pos = 0
        for idx, pad_pos in enumerate(self.pad_positions):
            parts.append(self.template_ids[current_pos:pad_pos])
            placeholder_key = self.placeholders[idx]
            parts.append(truncated[placeholder_key])
            current_pos = pad_pos + 1
        if current_pos < len(self.template_ids):
            parts.append(self.template_ids[current_pos:])
        return list(itertools.chain(*parts))

    def _truncate_args(self, **kwargs):
        remaining_length = self.max_length - len(self.template_ids) + len(self.pad_positions)
        values = sorted([(key, val) for key, val in kwargs.items()], key=lambda v: len(v[1]))
        truncated = {}
        for position, pair in enumerate(values):
            key, val = pair
            max_len_per_value = int(remaining_length / (len(values) - position))
            if len(val) > max_len_per_value:
                val = val[0:max_len_per_value]
            remaining_length -= len(val)
            truncated[key] = val
        return truncated
