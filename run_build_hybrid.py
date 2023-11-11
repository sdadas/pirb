import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict
from transformers import HfArgumentParser
from utils.system import set_java_env
set_java_env()
from data import BEIRTask, IndexInput
from search import SearchIndex, XGBRankerHybrid, HybridIndex, AutoIndex


@dataclass
class BuildHybridArgs:
    models_config: str = field(
        metadata={"help": "List of models to be used for training hybrid index"},
    )
    output_name: str = field(
        metadata={"help": "Name of the hybrid model"},
    )
    data_dir: str = field(
        default="data_train",
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
    k: int = field(
        default=100,
        metadata={"help": "Maximum number of hits"},
    )
    max_train_samples: int = field(
        default=50_000,
        metadata={"help": "Max samples for training, use 0 to load all data"},
    )


class HybridBuilder:

    def __init__(self, args: BuildHybridArgs):
        self.args = args
        self.task = BEIRTask("msmarco-pl", splits=("train",))
        self.task.prepare_task(self.args.data_dir)
        self.indices: List[SearchIndex] = self._create_indices()

    def _create_indices(self):
        with open(self.args.models_config, "r", encoding="utf-8") as config_file:
            model_specs: List[Dict] = json.load(config_file)
            assert isinstance(model_specs, list)
        return [AutoIndex.from_config(spec, self.task, self.args) for spec in model_specs]

    def build(self):
        cache_prefix = f"train_{self.task.task_id}"
        queries: List[IndexInput] = list(self.task.queries(self.args.data_dir))
        if self.args.max_train_samples > 0:
            queries = queries[:self.args.max_train_samples]
        index_results: List[Dict] = []
        for index in self.indices:
            logging.info(f"Processing index {index.name()}")
            if not index.exists() and not index.results_exist(self.args.k, cache_prefix):
                index.build(self.task.passages(self.args.data_dir))
            results = index.search(queries, self.args.k, cache_prefix=cache_prefix)
            index_results.append(results)
        ranker = XGBRankerHybrid()
        ranker.fit(queries, index_results)
        hindex = HybridIndex(self.args.data_dir, self.args.output_name, self.indices, self.args.k, ranker, self.task)
        result = hindex.model_dict()
        output_path = f"config/{self.args.output_name}.json"
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump([result], output_file, ensure_ascii=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([BuildHybridArgs])
    args = parser.parse_args_into_dataclasses()[0]
    builder = HybridBuilder(args)
    builder.build()
