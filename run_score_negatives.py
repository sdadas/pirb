import json
import logging
import os.path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from tqdm import tqdm
from transformers import HfArgumentParser

from data import RawJsonlTask, IndexInput
from search import SearchIndex, AutoIndex, RerankerHybrid


@dataclass
class HardNegsArgs:
    retrievers: str = field(
        metadata={"help": "A config file containing list of retrieval models"},
    )
    retrievers_keys: str = field(
        metadata={"help": "Keys for retrieval models"},
    )
    reranker: str = field(
        metadata={"help": "A config file containing reranker definition"},
    )
    task_name: str = field(
        metadata={"help": "Name of the input directory containing queries and passages"},
    )
    output_name: str = field(
        metadata={"help": "Name of the output file"},
    )
    data_dir: str = field(
        default="data",
        metadata={"help": "Directory where task data is stored"},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Directory where indexes and cached results are stored"},
    )
    threads: int = field(
        default=8,
        metadata={"help": "Number of threads to use for indexing and searching Lucene"},
    )
    use_bettertransformer: bool = field(
        default=True,
        metadata={"help": "Patch dense encoders with BetterTransformer optimizations"},
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


class HardNegsBuilder:

    def __init__(self, args: HardNegsArgs):
        self.args = args
        self.task = RawJsonlTask(self.args.task_name)
        self.retriever_keys = [val.strip() for val in self.args.retrievers_keys.split(",")]
        self.indices: List[SearchIndex] = self._create_indices()
        assert len(self.retriever_keys) == len(self.indices)
        self.reraker: Optional[RerankerHybrid] = None

    def _create_indices(self):
        with open(self.args.retrievers, "r", encoding="utf-8") as config_file:
            model_specs: List[Dict] = json.load(config_file)
            assert isinstance(model_specs, list)
        return [AutoIndex.from_config(spec, self.task, self.args) for spec in model_specs]

    def _create_reranker(self):
        with open(self.args.reranker, "r", encoding="utf-8") as config_file:
            conf = json.load(config_file)
        return RerankerHybrid(**conf)

    def build(self):
        cache_prefix = self.task.task_id
        passages: Dict[str, str] = {val.id : val.text for val in self.task.passages(self.args.data_dir)}
        queries: List[IndexInput] = list(self.task.queries(self.args.data_dir))
        if self.args.queries_end_idx is None:
            self.args.queries_end_idx = len(queries)
        queries = queries[self.args.queries_start_idx:self.args.queries_end_idx]
        logging.info(f"Running retrieval for {len(queries)} queries")
        index_results: List[Dict] = []
        for index in self.indices:
            logging.info(f"Processing index {index.name()}")
            if not index.exists():
                index.build(self.task.passages(self.args.data_dir))
            results = index.search(queries, self.args.max_negs_per_index, cache_prefix=cache_prefix)
            index_results.append(results)
        self.reraker = self._create_reranker()
        output_path = os.path.join(self.args.data_dir, self.task.task_id, self.args.output_name)
        with open(output_path, "w", encoding="utf-8") as output_file:
            for query in tqdm(queries, desc="Scoring docs for queries"):
                out = self._process_query(query, passages, index_results)
                output_file.write(json.dumps(out, ensure_ascii=False))
                output_file.write("\n")

    def _process_query(self, query: IndexInput, passages: Dict, index_results: List[Dict]):
        negs = {}
        out = {"query_id": query.id, "positive": query.relevant, "negative": negs}
        ids2docs = lambda ids: [passages.get(val) for val in ids]
        unique_docs = set(query.relevant)
        out["positive_scores"] = self.reraker.rerank(query.text, ids2docs(query.relevant), proba=True)
        for idx, result in enumerate(index_results):
            hits = result.get(query.id)
            docids = [hit.id for hit in hits if hit.id not in unique_docs]
            key = self.retriever_keys[idx]
            unique_docs.update(docids)
            scores = self.reraker.rerank(query.text, ids2docs(docids), proba=True) if len(docids) > 0 else []
            negs[key] = {"docs": docids, "scores": scores}
        return out


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([HardNegsArgs])
    args = parser.parse_args_into_dataclasses()[0]
    builder = HardNegsBuilder(args)
    builder.build()
