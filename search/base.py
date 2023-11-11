import gzip
import json
import os.path
from abc import ABC
from typing import List, Iterable, Dict, Optional
import faiss
import numpy as np
import torch
from joblib import dump
from sentence_transformers import SentenceTransformer
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

    def model_dict(self) -> Dict:
        raise NotImplementedError()