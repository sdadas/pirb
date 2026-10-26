import json
import logging
import os.path
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sklearn.metrics import ndcg_score
from search.hybrid import RerankerHybrid
from tqdm import tqdm
from transformers import HfArgumentParser
from data import RawJsonlTask, IndexInput, RetrievalTask, IndexResult
from search import SearchIndex, AutoIndex


@dataclass
class HardNegsArgs:
    retrievers: str = field(
        metadata={"help": "A config file containing list of retrieval models"},
    )
    retrievers_keys: str = field(
        metadata={"help": "Keys for retrieval models"},
    )
    reranker: str = field(
        metadata={"help": "A reranker definition"},
    )
    task_name: str = field(
        metadata={"help": "Name of the input directory containing queries and passages"},
    )
    output_name: str = field(
        metadata={"help": "Name of the output file"},
    )
    data_dir: str = field(
        default="data2",
        metadata={"help": "Directory where task data is stored"},
    )
    cache_dir: str = field(
        default="cache2",
        metadata={"help": "Directory where indexes and cached results are stored"},
    )
    max_negs_per_index: int = field(
        default=32,
        metadata={"help": "Maximum number of negatives from each retriever"},
    )
    queries_start_idx: int = field(
        default=0,
        metadata={"help": "Queries staring index"},
    )
    queries_end_idx: int = field(
        default=None,
        metadata={"help": "Queries ending index"},
    )
    text_field: str = field(
        default="source",
        metadata={"help": "Name of the default text field"},
    )
    use_cached_scores: bool = field(
        default=True,
        metadata={"help": "Use scores.jsonl file if exists to get retrieved docs"},
    )
    proba: bool = field(
        default=True,
        metadata={"help": "Return probabilities"},
    )


class HardNegsBuilder:

    def __init__(self, args: HardNegsArgs):
        self.args = args
        self.task = RawJsonlTask(self.args.task_name)
        self.task.prepare_task(self.args.data_dir)
        self.retriever_keys = [val.strip() for val in self.args.retrievers_keys.split(",")]
        self.indices: List[SearchIndex]
        scores_path = os.path.join(self.args.data_dir, self.task.task_id, "scores.jsonl")
        if os.path.exists(scores_path) and self.args.use_cached_scores:
            self.indices = None
        else:
            self.indices = self.create_indices()
            assert len(self.retriever_keys) == len(self.indices)
        self.reraker: Optional[RerankerHybrid] = None

    def create_indices(self):
        with open(self.args.retrievers, "r", encoding="utf-8") as config_file:
            model_specs: List[Dict] = json.load(config_file)
            assert isinstance(model_specs, list)
        return [AutoIndex.from_config(spec, self.task, self.args) for spec in model_specs]

    def create_reranker(self):
        with open(self.args.reranker, "r", encoding="utf-8") as config_file:
            conf = json.load(config_file)
        self.reraker = RerankerHybrid(**conf)

    def build(self):
        passages: Dict[str, str] = {val.id : val.text for val in self.task.passages(self.args.data_dir)}
        queries: List[IndexInput] = list(self.task.queries(self.args.data_dir))
        if self.args.queries_end_idx is None:
            self.args.queries_end_idx = len(queries)
        queries = queries[self.args.queries_start_idx:self.args.queries_end_idx]
        scores_path = os.path.join(self.args.data_dir, self.task.task_id, "scores.jsonl")
        if os.path.exists(scores_path) and self.args.use_cached_scores:
            index_results = self._load_retrieved_docs_from_scores(scores_path, queries)
        else:
            index_results = self._run_retrievers(queries)
        self.create_reranker()
        output_path = os.path.join(self.args.data_dir, self.task.task_id, self.args.output_name)
        pbar = tqdm(total=len(queries))
        ndcg_count, ndcg_sum = 0, 0
        with open(output_path, "w", encoding="utf-8") as output_file:
            for query in queries:
                pbar.update(1)
                out = self.process_query(query, passages, index_results)
                output_file.write(json.dumps(out, ensure_ascii=False))
                output_file.write("\n")
                output_file.flush()
                ndcg = out["ndcg"]
                if ndcg is not None:
                    ndcg_count += 1
                    ndcg_sum += ndcg
                    pbar.set_postfix(ndcg=ndcg_sum / ndcg_count)
        pbar.close()

    def _run_retrievers(self, queries: List[IndexInput]):
        logging.info(f"Running retrieval for {len(queries)} queries")
        index_results: List[Dict] = []
        for index in self.indices:
            logging.info(f"Processing index {index.name()}")
            if not index.exists():
                index.build(self.task.passages(self.args.data_dir))
            results = index.search(queries, max(self.args.max_negs_per_index, 32))
            index_results.append(results)
        return index_results

    def _load_retrieved_docs_from_scores(self, file_path: str, queries: List[IndexInput]):
        logging.info(f"Loading retrieved docs from {file_path}")
        query_ids = {q.id for q in queries}
        results = defaultdict(dict)
        pbar = tqdm(total=Path(file_path).stat().st_size, unit='B', unit_scale=True, unit_divisor=1024, desc="scores")
        with open(file_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                pbar.update(len(line.encode("utf-8")))
                value = json.loads(line.strip())
                query_id = value["query_id"]
                if query_id not in query_ids:
                    continue
                negative: Dict = value["negative"]
                for system_name, system_scores in negative.items():
                    docs = system_scores["docs"]
                    scores = system_scores["scores"]
                    assert len(docs) == len(scores)
                    result = [IndexResult(key, val) for key, val in zip(docs, scores)]
                    results[system_name][query_id] = result
        pbar.close()
        return list(results.values())

    def process_query(self, query: IndexInput, passages: Dict, index_results: List[Dict]):
        negs = {}
        out = {"query_id": query.id, "positive": query.relevant, "negative": negs}
        ids2docs = lambda ids: [passages.get(val) for val in ids]
        unique_docs = set(query.relevant)
        out["positive_scores"] = self.reraker.rerank(query.text, ids2docs(query.relevant), proba=self.args.proba)
        for idx, result in enumerate(index_results):
            hits = result.get(query.id)
            if hits is None:
                hits = []
            if self.task.skip_self:
                hits = [hit for hit in hits if hit.id != query.id]
            docids = [hit.id for hit in hits if hit.id not in unique_docs]
            if len(docids) > self.args.max_negs_per_index:
                docids = docids[:self.args.max_negs_per_index]
            key = self.retriever_keys[idx]
            unique_docs.update(docids)
            scores = self.reraker.rerank(query.text, ids2docs(docids), proba=self.args.proba) if len(docids) > 0 else []
            negs[key] = {"docs": docids, "scores": scores}
        out["ndcg"] = self._compute_ndcg(out)
        return out

    def _compute_ndcg(self, doc: Dict, k: int = 10):
        relevant = doc["positive"]
        if len(relevant) == 0:
            return None
        relevant_set = set(doc["positive"])
        hits = {}
        for system, negs in doc["negative"].items():
            docids = negs["docs"]
            scores = negs["scores"]
            for idx in range(len(docids)):
                hits[docids[idx]] = scores[idx]
        for docid, score in zip(doc["positive"], doc["positive_scores"]):
            hits[docid] = score
        hits = [(docid, score) for docid, score in hits.items()]
        hits.sort(key=lambda v: v[1], reverse=True)
        if len(hits) > k:
            hits = hits[:k]
        not_relevant = [hit[0] for hit in hits if hit[0] not in relevant_set]
        all_docs = relevant + not_relevant
        y_true = [1.0] * len(relevant) + [0.0 for _ in range(len(not_relevant))]
        min_score = min((hit[1] for hit in hits)) if len(hits) > 0 else 0.0
        hit_scores = {hit[0]: hit[1] - min_score + 1e-6 for hit in hits}
        y_pred = [hit_scores.get(docid, 0.0) for docid in all_docs]
        pad = lambda a, i: a if len(a) > i else a + [0.0] * (i - len(a))
        y_true, y_pred = pad(y_true, k), pad(y_pred, k)
        return ndcg_score([y_true], [y_pred], k=k)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    parser = HfArgumentParser([HardNegsArgs])
    args = parser.parse_args_into_dataclasses()[0]
    RetrievalTask.TEXT_FIELD = args.text_field
    logging.info(f"Using {args.text_field} as the default text field")
    builder = HardNegsBuilder(args)
    builder.build()
