import json
import logging
import os.path
from dataclasses import dataclass, field
from typing import List, Any, Dict
from tqdm import tqdm
from transformers import HfArgumentParser
from utils.system import set_java_env
import xlsxwriter
import torch.cuda
set_java_env()

from data import RawJsonlTask, MFAQTask, RetrievalTask, BEIRTask, PolEvalTask, MAUPQATask, GPTExamsTask
from search import SearchIndex, AutoIndex


@dataclass
class CompareArgs:
    first_config: str = field(
        metadata={"help": "Config of the first model"},
    )
    second_config: str = field(
        metadata={"help": "Config of the first model"},
    )
    output_dir: str = field(
        metadata={"help": "Output directory"},
    )
    ndcg_k: int = field(
        default=10,
        metadata={"help": "Number of hits for NDCG score"},
    )
    recall_k: int = field(
        default=100,
        metadata={"help": "Number of hits for Recall score"},
    )
    threads: int = field(
        default=8,
        metadata={"help": "Number of threads to use for indexing and searching Lucene"},
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
        default=True,
        metadata={"help": "Patch dense encoders with BetterTransformer optimizations"},
    )
    scope: str = field(
        default="full",
        metadata={"help": "Scope of benchmark to run: full, small (only datasets <100MB), tiny (only datasets <20MB)"},
    )
    raw_mode: bool = field(
        default=True,
        metadata={"help": "Use raw vector operations instead of FAISS index"},
    )
    query_limit: int = field(
        default=None,
        metadata={"help": "Maximum number of queries to use for evaluation on each dataset"}
    )
    rerank_limit: int = field(
        default=None,
        metadata={"help": "Maximum number of retrieved docs to sort using reranker (all docs are sorted by default)"}
    )
    limit_outputs: int = field(
        default=1000,
        metadata={"help": "Maximum number of rows to include in the output file"}
    )


@dataclass
class ScoreDiff:
    query: str
    doc: str
    rank1: int
    rank2: int

    def rank_diff(self):
        return self.rank1 - self.rank2

    def abs_diff(self):
        return abs(self.rank1 - self.rank2)


class ModelComparisonEvaluator:

    def __init__(self, args: CompareArgs):
        self.args: CompareArgs = args

    def compare(self, task: RetrievalTask, index1: SearchIndex, index2: SearchIndex):
        score_diffs = []
        passages = {val.id: val.text for val in task.passages(self.args.data_dir)}
        task.set_limit_queries(self.args.query_limit)
        results1 = self._get_results(index1)
        results2 = self._get_results(index2)
        queries = list(task.queries(self.args.data_dir))
        for query in tqdm(queries, desc="Comparing models"):
            hits1 = results1.get(query.id) or []
            hits2 = results2.get(query.id) or []
            if task.skip_self:
                hits1 = [hit for hit in hits1 if hit.id != query.id]
                hits2 = [hit for hit in hits2 if hit.id != query.id]
            for docid in query.relevant:
                rank1 = self._find_relevant_index(hits1, docid)
                rank2 = self._find_relevant_index(hits2, docid)
                sd = ScoreDiff(query.text, passages.get(docid), rank1, rank2)
                score_diffs.append(sd)
        self._write_output(task.task_id, score_diffs)

    def _get_results(self, index: SearchIndex):
        cache_prefix = task.task_id
        needs_rebuild = not index.exists() and not index.results_exist(self.args.recall_k, cache_prefix)
        task.set_limit_queries(self.args.query_limit)
        if needs_rebuild:
            index.build(task.passages(self.args.data_dir))
        queries = list(task.queries(self.args.data_dir))
        return index.search(queries, self.args.recall_k, cache_prefix=cache_prefix)

    def _find_relevant_index(self, hits: List[Any], docid: str):
        for idx, hit in enumerate(hits):
            if hit.id == docid:
                return idx
        return self.args.recall_k  # docid not found
    
    def _write_output(self, task_id: str, score_diffs: List[ScoreDiff]):
        score_diffs = sorted(score_diffs, key=lambda v: -v.abs_diff())
        score_diffs = score_diffs[:self.args.limit_outputs]
        score_diffs = sorted(score_diffs, key=lambda v: v.rank_diff())
        os.makedirs(self.args.output_dir, exist_ok=True)
        workbook = xlsxwriter.Workbook(os.path.join(self.args.output_dir, f"{task_id}.xlsx"))
        sheet = workbook.add_worksheet()
        header = ["Diff", "Abs Diff", "Query", "Document", "Rank1", "Rank2"]
        sheet.write_row(0, 0, header)
        for idx, diff in enumerate(score_diffs):
            row = [diff.rank_diff(), diff.abs_diff(), diff.query, diff.doc, diff.rank1, diff.rank2]
            sheet.write_row(idx + 1, 0, row)
        workbook.close()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as config_file:
        definition: List[Dict] = json.load(config_file)
        if isinstance(definition, list):
            return definition[0]
        else:
            return definition


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([CompareArgs])
    args = parser.parse_args_into_dataclasses()[0]

    benchmark = [
        # Web crawled datasets
        RawJsonlTask("gemini"),
        RawJsonlTask("odi"),
        RawJsonlTask("onet"),
        RawJsonlTask("zapytajfizyka"),
        RawJsonlTask("techpedia"),
        RawJsonlTask("pwn"),
        RawJsonlTask("eprawnik"),
        RawJsonlTask("specprawnik"),
        RawJsonlTask("abczdrowie"),
        # MAUPQA datasets
        MAUPQATask("1z10"),
        MAUPQATask("czy-wiesz-v2"),
        MAUPQATask("gpt3-cc"),
        MAUPQATask("gpt3.5-cc"),
        MAUPQATask("gpt3.5-wiki"),
        MAUPQATask("mkqa"),
        MAUPQATask("mqa"),
        MAUPQATask("multilingual-NLI"),
        MAUPQATask("poleval2021-pairs"),
        MAUPQATask("poquad"),
        MAUPQATask("templates"),
        MAUPQATask("wiki-def"),
        # PolEval-2022
        PolEvalTask("dev-0", "wiki-trivia"),
        PolEvalTask("test-A", "wiki-trivia"),
        PolEvalTask("test-A", "legal-questions"),
        PolEvalTask("test-A", "allegro-faq"),
        PolEvalTask("test-B", "wiki-trivia"),
        PolEvalTask("test-B", "legal-questions"),
        PolEvalTask("test-B", "allegro-faq"),
        # BEIR-PL datasets
        BEIRTask("arguana-pl", skip_self=True),
        BEIRTask("dbpedia-pl"),
        BEIRTask("fiqa-pl"),
        BEIRTask("hotpotqa-pl"),
        BEIRTask("msmarco-pl", splits=("validation",)),
        BEIRTask("nfcorpus-pl"),
        BEIRTask("nq-pl"),
        BEIRTask("quora-pl", skip_self=True),
        BEIRTask("scidocs-pl"),
        BEIRTask("scifact-pl"),
        BEIRTask("trec-covid-pl"),
        # Other datasets
        MFAQTask(),
        GPTExamsTask()
    ]
    os.makedirs(args.data_dir, exist_ok=True)
    benchmark = [task for task in benchmark if task.is_available(args.data_dir)]
    for task in benchmark:
        task.prepare_task(args.data_dir)

    if args.scope in ("small", "tiny"):
        benchmark = [task for task in benchmark if task.is_small(args.data_dir, args.scope)]
    elif args.scope != "full":
        filer_ds = {args.scope.lower()} if "," not in args.scope else set(
            [val.strip().lower() for val in args.scope.split(",")])
        benchmark = [task for task in benchmark if task.task_id.lower() in filer_ds]

    logging.info("Running comparison on %d datasets", len(benchmark))
    encoder1 = load_config(args.first_config)
    encoder2 = load_config(args.second_config)
    evaluator = ModelComparisonEvaluator(args)
    for task in benchmark:
        print(f"Results for task: {task.task_id.upper()}")
        model1 = AutoIndex.from_config(encoder1, task, args)
        model2 = AutoIndex.from_config(encoder2, task, args)
        evaluator.compare(task, model1, model2)
        del model1, model2
        torch.cuda.empty_cache()
