import json
import logging
import os.path
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Any, Dict
import torch
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import HfArgumentParser
from utils.system import set_java_env
set_java_env()

from data import RetrievalTask, Benchmark, IndexResult
from search import SearchIndex, AutoIndex

# patch fcntl on Windows
try:
    import fcntl
except ImportError:
    class fcntl:
        LOCK_EX: int = 0
        LOCK_UN: int = 1
        def flock(self, *args):
            pass


@dataclass
class BenchmarkArgs:
    models_config: str = field(
        metadata={"help": "Path to models config file"},
    )
    benchmark_config: str = field(
        default="config/benchmarks/pirb.json",
        metadata={"help": "Path to the benchmark config file"},
    )
    ndcg_k: int = field(
        default=10,
        metadata={"help": "Number of hits for NDCG score"},
    )
    recall_k: int = field(
        default=100,
        metadata={"help": "Number of hits for Recall score"},
    )
    data_dir: str = field(
        default="data",
        metadata={"help": "Directory where task data is stored"},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Directory where indexes and cached results are stored"},
    )
    use_bettertransformer: bool = field(
        default=False,
        metadata={"help": "Patch dense encoders with BetterTransformer optimizations"},
    )
    scope: str = field(
        default="full",
        metadata={"help": "Scope of benchmark to run: full, small (only datasets <100MB), tiny (only datasets <20MB)"},
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite any cache or results even if exists"}
    )
    query_limit: int = field(
        default=None,
        metadata={"help": "Maximum number of queries to use for evaluation on each dataset"}
    )
    rerank_limit: int = field(
        default=None,
        metadata={"help": "Maximum number of retrieved docs to sort using reranker (all docs are sorted by default)"}
    )
    rm: bool = field(
        default=False,
        metadata={"help": "Remove cached indexes after evaluation"}
    )


class RetrievalEvaluator:

    def __init__(self, args: BenchmarkArgs):
        self.args: BenchmarkArgs = args
        self.stats = {}

    def eval_task(self, task: RetrievalTask, index: SearchIndex, metadata=None):
        cache_prefix = task.task_id
        needs_rebuild = not index.exists() and not index.results_exist(self.args.recall_k, cache_prefix)
        task.set_limit_queries(self.args.query_limit)
        if self.args.overwrite or needs_rebuild:
            index.build(task.passages(self.args.data_dir))
        chunk_doc_mapping = task.get_chunk_document_mapping(self.args.data_dir)
        queries = list(task.queries(self.args.data_dir))
        results = index.search(queries, self.args.recall_k, cache_prefix=cache_prefix, overwrite=self.args.overwrite)
        mrr_sum, total = 0.0, 0
        acc_correct = Counter()
        ndcg_sum = 0.0
        recall_sum = 0.0
        for query in tqdm(queries, desc="Computing evaluation metrics"):
            hits = results.get(query.id)
            if hits is None: hits = []
            hits = self._map_chunks_to_docs(hits, chunk_doc_mapping)
            if task.skip_self:
                hits = [hit for hit in hits if hit.id != query.id]
            docids = [hit.id for hit in hits]
            total += 1
            relevant_set = set(query.relevant)
            found_idx = -1
            for idx, docid in enumerate(docids):
                if idx >= self.args.ndcg_k:
                    break
                if docid in relevant_set:
                    found_idx = idx
                    break
            for i in range(5):
                acc_correct[i] += 1 if i >= found_idx >= 0 else 0
            mrr_sum += (1 / (found_idx + 1)) if found_idx >= 0 else 0
            y_true, y_pred = self._get_doc_rankings(hits, query.relevant, query.relevant_scores, self.args.ndcg_k)
            ndcg_sum += ndcg_score([y_true], [y_pred], k=self.args.ndcg_k)
            recall_sum += self._recall(hits, query.relevant)
        ndcg = (ndcg_sum / total) * 100.0
        recall = (recall_sum / total) * 100.0
        mrr = 100 * mrr_sum / total
        metrics = {
            "Accuracy@1": 100 * acc_correct[0] / total,
            "Accuracy@1-5": [100 * acc_correct[i] / total for i in range(5)],
            f"Recall@{self.args.recall_k}": recall,
            f"MRR@{self.args.ndcg_k}": mrr,
            f"NDCG@{self.args.ndcg_k}": ndcg,
        }
        self._log_score(task, index, metrics, metadata)
        self.stats["queries"] = self.stats.get("queries", 0) + len(queries)
        index.accumulate_stats(self.stats, task)
        print(", ".join([f"{k}: {v:.4f}%" for k, v in metrics.items() if not isinstance(v, list)]) + f" ({index.name()})")
        return ndcg

    def _map_chunks_to_docs(self, hits: List[IndexResult], mapping: Dict):
        if mapping is None:
            return hits
        doc_ids = set()
        results = []
        for hit in hits:
            doc_id = mapping.get(hit.id, hit.id)
            if doc_id not in doc_ids:
                results.append(IndexResult(doc_id, hit.score))
                doc_ids.add(doc_id)
        return results

    def _get_doc_rankings(self, hits: List[Any], relevant: List[str], relevant_scores: List, k: int):
        if len(hits) > k:
            hits = hits[:k]
        relevant_set = set(relevant)
        not_relevant = [hit.id for hit in hits if hit.id not in relevant_set]
        all_docs = relevant + not_relevant
        y_true = [float(val) for val in relevant_scores] + [0.0 for _ in range(len(not_relevant))]
        min_score = min((hit.score for hit in hits)) if len(hits) > 0 else 0.0
        hit_scores = {hit.id: hit.score - min_score + 1e-6 for hit in hits}
        y_pred = [hit_scores.get(docid, 0.0) for docid in all_docs]
        pad = lambda a, i: a if len(a) > i else a + [0.0] * (i - len(a))
        return pad(y_true, k), pad(y_pred, k)

    def _recall(self, hits: List[Any], relevant: List[str]):
        relevant_set = set(relevant)
        found = 0
        for hit in hits:
            if hit.id in relevant_set:
                found += 1
        return min(found / len(relevant_set), 1.0)

    def _log_score(self, task: RetrievalTask, index: SearchIndex, metrics: Dict, metadata: Dict):
        res = {"task_id": task.task_id, "group_id": task.group_id, "model": index.name(),"metrics": metrics}
        if "strategy" in metadata.keys():
            metadata = dict(metadata)
            metadata["strategy"] = {"type": metadata["strategy"]["type"]}
        if metadata is not None:
            res["metadata"] = metadata
        res["timestamp"] = datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
        with open("runlog.txt", "a", encoding="utf-8") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            json.dump(res, output_file, ensure_ascii=False)
            output_file.write("\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)


def _load_models(config_path: str) -> List[Dict]:
    if config_path.endswith("*"):
        config_path = config_path[:-1]
        files = sorted(os.listdir(config_path))
        encoders = []
        for file in files:
            if file.lower().endswith(".json"):
                encoders.extend(_load_models(os.path.join(config_path, file)))
        return encoders
    else:
        with open(config_path, "r", encoding="utf-8") as config_file:
            encoders: List[Dict] = json.load(config_file)
            assert isinstance(encoders, list), "--models_config should contain a list of encoder models"
            return encoders


def _add_index_path(paths: List, index):
    if index.__class__.__name__ == "HybridIndex":
        for idx in index.indices:
            paths.append(idx.index_path())
    else:
        paths.append(index.index_path())


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    parser = HfArgumentParser([BenchmarkArgs])
    args = parser.parse_args_into_dataclasses()[0]
    encoders = _load_models(args.models_config)
    logging.info("Evaluating %d models: %s", len(encoders), ", ".join([v["name"] for v in encoders]))

    benchmark = Benchmark.from_config(args.benchmark_config)
    os.makedirs(args.data_dir, exist_ok=True)
    benchmark.prepare(args.data_dir)
    benchmark = benchmark.filter(args.scope, args.data_dir)

    logging.info("Running evaluation on %d datasets", len(benchmark))
    evaluator = RetrievalEvaluator(args)
    index_paths = []
    for encoder in encoders:
        ndcg_tasks = 0.0
        for task in benchmark.tasks:
            print(f"Results for task: {task.task_id.upper()}")
            model = AutoIndex.from_config(encoder, task, args)
            _add_index_path(index_paths, model)
            ndcg_tasks += evaluator.eval_task(task, model, metadata=encoder)
            del model
            torch.cuda.empty_cache()
        print(evaluator.stats)
        evaluator.stats = {}
        print(f"Average NDCG@{args.ndcg_k} for {len(benchmark)} tasks: {ndcg_tasks / len(benchmark):.2f}")
    if args.rm:
        logging.info("Removing cached indexes")
        for index_path in index_paths:
            if index_path is not None and os.path.isdir(index_path):
                logging.info(index_path)
                shutil.rmtree(index_path)
