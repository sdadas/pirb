import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict
from tqdm import tqdm
from transformers import HfArgumentParser
from data import IndexInput, RawJsonlTask, IndexResult
from search import AutoIndex, SearchIndex
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


@dataclass
class PrepareRagArgs:
    model_config: str = field(
        metadata={"help": "A config file containing retrieval model"},
    )
    queries_name: str = field(
        metadata={"help": "Name of the generated queries file"},
    )
    output_name: str = field(
        metadata={"help": "Name of the output file"},
    )
    task_name: str = field(
        metadata={"help": "Name of the input directory containing queries and passages"},
    )
    data_dir: str = field(
        default="data",
        metadata={"help": "Directory where task data is stored"},
    )
    cache_dir: str = field(
        default="cache2",
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
    queries_start_idx: int = field(
        default=0,
        metadata={"help": "Queries staring index"},
    )
    queries_end_idx: int = field(
        default=None,
        metadata={"help": "Queries ending index"},
    )
    top_k: int = field(
        default=10,
        metadata={"help": "How many retrieved results should be included in the output"},
    )


class PrepareRagBuilder:

    def __init__(self, args: PrepareRagArgs):
        self.args = args
        self.task = RawJsonlTask(self.args.task_name)
        self.queries: List[IndexInput] = self._load_generated_queries()
        self.index: SearchIndex = self._create_index()

    def _create_index(self):
        with open(self.args.model_config, "r", encoding="utf-8") as config_file:
            model_spec = json.load(config_file)
            if isinstance(model_spec, list):
                model_spec = model_spec[0]
        return AutoIndex.from_config(model_spec, self.task, self.args)

    def _load_generated_queries(self):
        queries_path = os.path.join(self.args.data_dir, self.args.task_name, self.args.queries_name)
        results = {}
        with open(queries_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                value = json.loads(line.strip())
                question = value["contents"]
                other = results.get(question.lower(), None)
                if other is not None:
                    for docid in value["relevant"]:
                        other.relevant.append(docid)
                else:
                    results[question.lower()] = IndexInput(value["id"], question, value["relevant"])
        results = list(results.values())
        results.sort(key=lambda v: v.id)
        if self.args.queries_end_idx is None:
            self.args.queries_end_idx = len(results)
        results = results[self.args.queries_start_idx:self.args.queries_end_idx]
        return results

    def build(self):
        output_path = os.path.join(self.args.data_dir, self.task.task_id, self.args.output_name)
        logging.info(f"Running retrieval for {len(self.queries)} queries")
        if not self.index.exists():
            self.index.build(self.task.passages(self.args.data_dir))
        results = self.index.search(self.queries, k=self.args.top_k)
        pbar = tqdm(total=len(self.queries))
        with open(output_path, "w", encoding="utf-8") as output_file:
            for query in self.queries:
                pbar.update(1)
                hits = results.get(query.id)
                if hits is None: hits = []
                out = self.process_query(query, hits)
                output_file.write(json.dumps(out, ensure_ascii=False))
                output_file.write("\n")
                output_file.flush()
        pbar.close()

    def process_query(self, query: IndexInput, hits: List[IndexResult]):
        retrieved = [hit.id for hit in hits]
        retrieved_scores = [hit.score for hit in hits]
        return {
            "id": query.id,
            "contents": query.text,
            "relevant": query.relevant,
            "retrieved": retrieved,
            "retrieved_scores": retrieved_scores,
            "retrieved_relevant": any([(docid in retrieved) for docid in query.relevant])
        }


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([PrepareRagArgs])
    args = parser.parse_args_into_dataclasses()[0]
    builder = PrepareRagBuilder(args)
    builder.build()
