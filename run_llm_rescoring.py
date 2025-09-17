import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import HfArgumentParser
from data import RawJsonlTask, RetrievalTask
from search.reranking import RankGPTReranker


@dataclass
class RescoringArgs:
    llm_config: str = field(
        metadata={"help": "Path to the LLM config JSON file"},
    )
    task_name: str = field(
        metadata={"help": "Name of the input directory containing queries and passages"},
    )
    output_name: str = field(
        default="scores_rescored.jsonl",
        metadata={"help": "Name of the output file"},
    )
    data_dir: str = field(
        default="data2",
        metadata={"help": "Directory where task data is stored"},
    )
    num_docs: int = field(
        default=20,
        metadata={"help": "Number of top documents to rerank"},
    )
    limit: int = field(
        default=None,
        metadata={"help": "Max number of rows to process"},
    )
    text_field: str = field(
        default="source",
        metadata={"help": "Name of the default text field"},
    )
    update_better_only: bool = field(
        default=True,
        metadata={"help": "Change the original ranking only if the new ranking is better"},
    )


@dataclass
class LLMRescorerStats:
    rows: int = 0
    updated_rows: int = 0
    ndcg_before: float = 0
    ndcg_after: float = 0


class LLMRescorer:

    def __init__(self, args: RescoringArgs):
        self.args = args
        self.task = RawJsonlTask(self.args.task_name)
        self.task.prepare_task(self.args.data_dir)
        self.queries = {query.id: query.text for query in self.task.queries(self.args.data_dir)}
        self.passages = {passage.id: passage.text for passage in self.task.passages(self.args.data_dir)}
        self.model = self._create_model()

    def _create_model(self):
        assert os.path.exists(self.args.llm_config), f"Config file {self.args.llm_config} does not exist"
        with open(self.args.llm_config, "r", encoding="utf-8") as input_file:
            config = json.load(input_file)
        return RankGPTReranker(**config)

    def rescore(self):
        scores_path = os.path.join(self.args.data_dir, self.task.task_id, "scores.jsonl")
        output_path = os.path.join(self.args.data_dir, self.task.task_id, self.args.output_name)
        assert os.path.exists(scores_path), f"Scores file {scores_path} not found"
        processed_ids = set()
        if os.path.exists(output_path):
            processed_ids = self._get_processed_ids(output_path)
        scores_size = Path(scores_path).stat().st_size
        pbar = tqdm(total=scores_size, unit="B", unit_scale=True, unit_divisor=1024, desc="rescoring")
        chunk = []
        stats = LLMRescorerStats()
        with open(scores_path, "r", encoding="utf-8") as scores_file, open(output_path, "a", encoding="utf-8") as out:
            for idx, line in enumerate(scores_file):
                if self.args.limit is not None and idx >= self.args.limit:
                    break
                pbar.update(len(line.encode("utf-8")))
                value = json.loads(line.strip())
                query_id = value["query_id"]
                if query_id in processed_ids:
                    continue
                ndcg = value.get("ndcg", None)
                if self.args.update_better_only and (ndcg is None or ndcg == 1.0):
                    out.write(json.dumps(value, ensure_ascii=False) + "\n")  # passthrough
                    if ndcg is not None:
                        stats.ndcg_before += ndcg
                        stats.ndcg_after += ndcg
                        stats.rows += 1
                elif len(chunk) < self.model.batch_size:
                    chunk.append(value)
                else:
                    self._rescore_chunk(chunk, stats, out, pbar)
                    chunk = [value]
            self._rescore_chunk(chunk, stats, out, pbar)
        pbar.close()

    def _get_processed_ids(self, output_path: str):
        query_ids = set()
        with open(output_path, "r", encoding="utf-8") as output_file:
            for line in output_file:
                value = json.loads(line.strip())
                query_ids.add(value["query_id"])
        return query_ids

    def _rescore_chunk(self, chunk: List[Dict], stats: LLMRescorerStats, out: TextIO, pbar):
        if len(chunk) == 0:
            return
        queries, ids, texts, scores = [], [], [], []
        for row in chunk:
            query, doc_ids, doc_texts, doc_scores = self._prepare_input(row)
            queries.append(query)
            ids.append(doc_ids)
            texts.append(doc_texts)
            scores.append(doc_scores)
        sorted_texts = self.model.rerank_many_slices(queries, texts)
        for idx in range(len(queries)):
            row = chunk[idx]
            output_row = self._apply_output(row, ids[idx], texts[idx], scores[idx], sorted_texts[idx], queries[idx])
            ndcg_before = self._compute_ndcg(row)
            ndcg_after = self._compute_ndcg(output_row)
            if ndcg_before is not None and ndcg_after is not None:
                output_row["ndcg"] = ndcg_after
                stats.rows += 1
                stats.ndcg_before += ndcg_before
                if ndcg_after < ndcg_before and self.args.update_better_only:
                    stats.ndcg_after += ndcg_before
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    stats.updated_rows += 1
                    stats.ndcg_after += ndcg_after
                    out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            else:
                if self.args.update_better_only:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    stats.updated_rows += 1
                    out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        ndcg_before = stats.ndcg_before / stats.rows
        ndcg_after = stats.ndcg_after / stats.rows
        ndcg_diff = ndcg_after - ndcg_before
        pbar.set_postfix(
            ndcg_before=ndcg_before,
            ndcg_after=ndcg_after,
            ndcg_diff=ndcg_diff,
            rows=stats.rows,
            updated_rows=stats.updated_rows
        )

    def _prepare_input(self, row: Dict):
        query = self.queries[row["query_id"]]
        passages: List[Tuple] = []
        for docid, score in zip(row["positive"], row["positive_scores"]):
            passages.append((docid, score))
        for model_id, results in row["negative"].items():
            for docid, score in zip(results["docs"], results["scores"]):
                passages.append((docid, score))
        passages.sort(key=lambda x: x[1], reverse=True)
        passages = passages[:self.args.num_docs]
        doc_ids = [val[0] for val in passages]
        doc_scores = [val[1] for val in passages]
        doc_texts = [self.passages[docid] for docid in doc_ids]
        return query, doc_ids, doc_texts, doc_scores

    def _apply_output(self, row: Dict, ids: List[str], texts: List[str], scores: List[float], sorted_texts: List[str], query: str):
        text2id = {text: docid for docid, text in zip(ids, texts)}
        sorted_ids = [text2id[text] for text in sorted_texts]
        id2score = {docid: score for docid, score in zip(sorted_ids, scores)}
        output_row = copy.deepcopy(row)
        for idx, docid in enumerate(output_row["positive"]):
            if docid in id2score:
                output_row["positive_scores"][idx] = id2score[docid]
        for model_id, results in output_row["negative"].items():
            for idx, docid in enumerate(results["docs"]):
                if docid in id2score:
                    results["scores"][idx] = id2score[docid]
        return output_row

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
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = HfArgumentParser([RescoringArgs])
    rescoring_args = parser.parse_args_into_dataclasses()[0]
    RetrievalTask.TEXT_FIELD = rescoring_args.text_field
    logging.info(f"Using {rescoring_args.text_field} as the default text field")
    rescorer = LLMRescorer(rescoring_args)
    rescorer.rescore()
