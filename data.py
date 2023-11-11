import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Iterable, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class IndexInput:
    id: str
    text: str
    relevant: Optional[List] = None
    relevant_scores: Optional[List] = None


@dataclass
class IndexResult:
    id: str
    score: float


class RetrievalTask:

    def __init__(self, task_id: str, group_id: str, skip_self: bool = False):
        self.task_id = task_id
        self.group_id = group_id
        self.skip_self = skip_self
        self.limit_queries = None

    def is_small(self, data_dir: str, scope: str) -> bool:
        max_size = {"small": 104_857_600, "tiny": 20_971_520}[scope]
        size = Path(self.passages_path(data_dir)).stat().st_size + Path(self.queries_path(data_dir)).stat().st_size
        return size <= max_size

    def task_cache_name(self):
        return self.task_id

    def prepare_task(self, data_dir: str):
        raise NotImplementedError()

    def passages_path(self, data_dir: str):
        return os.path.join(data_dir, self.task_id, "passages/passages.jsonl")

    def queries_path(self, data_dir: str):
        return os.path.join(data_dir, self.task_id, "queries/queries.jsonl")

    def passages(self, data_dir: str, verbose: bool = True) -> Iterable[IndexInput]:
        input_path = self.passages_path(data_dir)
        with open(input_path, "r", encoding="utf-8") as input_file:
            for line in tqdm(input_file, desc="Reading passages", disable=not verbose):
                value = json.loads(line.strip())
                yield IndexInput(value["id"], value["contents"])

    def queries(self, data_dir: str) -> Iterable[IndexInput]:
        input_path = self.queries_path(data_dir)
        with open(input_path, "r", encoding="utf-8") as queries_file:
            for idx, line in enumerate(queries_file):
                if self.limit_queries is not None and idx >= self.limit_queries:
                    break
                value = json.loads(line.strip())
                relevant = value["relevant"]
                relevant_scores = value.get("relevant_scores", None)
                if relevant_scores is None:
                    relevant_scores = [1] * len(relevant)
                yield IndexInput(value["id"], value["contents"], relevant, relevant_scores)

    def exists(self, data_dir):
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        Path(passages_path).parent.mkdir(parents=True, exist_ok=True)
        Path(queries_path).parent.mkdir(parents=True, exist_ok=True)
        return os.path.exists(passages_path) and os.path.exists(queries_path)

    def write_file(self, output_path: str, data: List):
        with open(output_path, "w", encoding="utf-8") as outfile:
            for val in data:
                outfile.write(json.dumps(val, ensure_ascii=False))
                outfile.write("\n")


class RawJsonlTask(RetrievalTask):

    def __init__(self, task_id: str):
        super().__init__(task_id, "Web")

    def prepare_task(self, data_dir: str):
        passages_path = os.path.join(data_dir, self.task_id, "passages/passages.jsonl")
        queries_path = os.path.join(data_dir, self.task_id, "queries/queries.jsonl")
        if not os.path.exists(passages_path):
            raise ValueError("Missing passages file %s, you should copy it manually" % (passages_path,))
        if not os.path.exists(queries_path):
            raise ValueError("Missing queries file %s, you should copy it manually" % (queries_path,))


class MFAQTask(RetrievalTask):

    def __init__(self):
        super().__init__("mfaq", "Other")

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        ds = load_dataset("clips/mfaq", name="pl_flat", split="train")
        with open(passages_path, "w", encoding="utf-8") as p, open(queries_path, "w", encoding="utf-8") as q:
            for idx, row in enumerate(ds):
                question = row.get("question").replace("\n", " ").replace("\r", " ")
                answer = row.get("answer").replace("\n", " ").replace("\r", " ")
                query = {"id": f"q_{idx + 1}", "contents": question, "relevant": [f"p_{idx + 1}"]}
                q.write(json.dumps(query, ensure_ascii=False))
                q.write("\n")
                passage = {"id": f"p_{idx + 1}", "contents": answer}
                p.write(json.dumps(passage, ensure_ascii=False))
                p.write("\n")


class BEIRTask(RetrievalTask):

    def __init__(self, task_id: str, skip_self: bool = False, splits=("test",)):
        super().__init__(task_id, "BEIR-PL", skip_self)
        self.splits: Iterable[str] = splits

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        logging.info("Preparing task %s", self.task_id)
        dataset_name = f"clarin-knext/{self.task_id}"
        self._write_passages(dataset_name, passages_path)
        self._write_queries(dataset_name, queries_path)

    def _write_passages(self, dataset_name: str, output_path: str):
        passages = load_dataset(dataset_name, name="corpus", split="corpus")
        with open(output_path, "w", encoding="utf-8") as out:
            for row in passages:
                title = row.get("title").replace("\n", " ").replace("\r", " ").strip()
                text = row.get("text").replace("\n", " ").replace("\r", " ")
                pid = row.get("_id")
                if len(title) > 0:
                    text = title + " " + text
                out.write(json.dumps({"id": pid, "contents": text}, ensure_ascii=False))
                out.write("\n")

    def _write_queries(self, dataset_name: str, output_path: str):
        queries = load_dataset(dataset_name, name="queries", split="queries")
        results = {}
        for row in queries:
            text = row.get("text").replace("\n", " ").replace("\r", " ")
            qid = row.get("_id")
            res = {"id": qid, "contents": text, "relevant": []}
            results[qid] = res
        qrels_dataset = dataset_name + "-qrels"
        for eval_split in self.splits:
            qrels = load_dataset(qrels_dataset, split=eval_split)
            for qrel in qrels:
                score = qrel.get("score")
                qid = str(qrel.get('query-id'))
                pid = str(qrel.get("corpus-id"))
                query = results.get(qid)
                if score > 0: query["relevant"].append((pid, score))
        total, dropped = 0, 0
        with open(output_path, "w", encoding="utf-8") as out:
            for key, val in results.items():
                relevant = sorted(val["relevant"], key=lambda v: -v[1])
                val["relevant"] = [qrel[0] for qrel in relevant]
                val["relevant_scores"] = [qrel[1] for qrel in relevant]
                total += 1
                if len(val["relevant"]) > 0:
                    assert len(val["relevant"]) == len(val["relevant_scores"])
                    out.write(json.dumps(val, ensure_ascii=False))
                    out.write("\n")
                else:
                    dropped += 1
        logging.info("Dropped %d of %d queries due to the empty relevant list", dropped, total)


class PolEvalTask(RetrievalTask):

    def __init__(self, split: str, domain: str):
        super().__init__(f"poleval-2022-{split}-{domain}", "PolEval-2022")
        self.split = split
        self.domain = domain

    def task_cache_name(self):
        return f"poleval-2022-{self.domain}"

    def passages_path(self, data_dir: str):
        return os.path.join(data_dir, "poleval-2022", self.domain, "passages/passages.jsonl")

    def queries_path(self, data_dir: str):
        return os.path.join(data_dir, "poleval-2022", self.domain, self.split, "queries.jsonl")

    def prepare_task(self, data_dir: str):
        domain_dir = os.path.join(data_dir, "poleval-2022", "raw", self.domain)
        split_dir = os.path.join(data_dir, "poleval-2022", "raw", self.split)
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        if os.path.exists(passages_path) and os.path.exists(queries_path):
            return
        logging.info("Preparing task %s", self.task_id)
        passages_path = self._fetch(self._passages_url(), os.path.join(domain_dir, "passages.jl"))
        self._write_passages(data_dir, passages_path)
        in_path = self._fetch(self._github_url("in.tsv"), os.path.join(split_dir, "in.tsv"))
        expected_path = self._fetch(self._github_url("expected.tsv"), os.path.join(split_dir, "expected.tsv"))
        self._write_queries(data_dir, in_path, expected_path)

    def _write_passages(self, data_dir: str, input_path: str):
        output_path = self.passages_path(data_dir)
        if os.path.exists(output_path):
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                value = json.loads(line.strip())
                text = " ".join(filter(lambda v: v is not None, [value.get("title"), value.get("text")]))
                doc = {"id": str(value["id"]), "contents": text}
                outfile.write(json.dumps(doc, ensure_ascii=False))
                outfile.write("\n")

    def _write_queries(self, data_dir: str, in_path: str, expected_path: str):
        output_path = self.queries_path(data_dir)
        if os.path.exists(output_path):
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        queries: List[Tuple] = []
        with open(in_path, "r", encoding="utf-8") as infile:
            for val in infile:
                fields = val.strip().split("\t")
                assert len(fields) == 2
                domain = fields[0]
                text = fields[1]
                queries.append((domain, text))
        with open(expected_path, "r", encoding="utf-8") as infile:
            expected = [val.strip().split("\t") for val in infile]
        assert len(queries) == len(expected)
        with open(output_path, "w", encoding="utf-8") as outfile:
            for idx, query in enumerate(queries):
                domain, text = query[0], query[1]
                if domain == self.domain:
                    relevant = expected[idx]
                    value = {"id": str(idx + 1), "contents": text, "relevant": relevant}
                    outfile.write(json.dumps(value, ensure_ascii=False))
                    outfile.write("\n")

    def _fetch(self, url: str, output_path: str):
        if os.path.exists(output_path):
            return output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            file_size = int(requests.head(url).headers["Content-Length"])
            progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=url.split('/')[-1])
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(8192)
            progress.close()
        return output_path

    def _github_url(self, file_name: str):
        return f"https://raw.githubusercontent.com/poleval/2022-passage-retrieval-secret/main/{self.split}/{file_name}"

    def _passages_url(self):
        return (f"https://huggingface.co/datasets/piotr-rybak/poleval2022-passage-retrieval-dataset/"
                f"resolve/main/{self.domain}/passages.jl")


class MAUPQATask(RetrievalTask):

    def __init__(self, subset: str):
        super().__init__(f"maupqa-{subset}", "MaupQA")
        self.subset = subset

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        logging.info("Preparing task %s", self.task_id)
        queries, passages = self.read_data()
        self.write_file(passages_path, passages)
        self.write_file(queries_path, queries)

    def read_data(self, include_negatives=False):
        queries = {}
        passages = {}
        pid = 0
        rows = load_dataset("ipipan/maupqa", name=self.subset, split="train")
        for row in rows:
            query_id = str(row["question_id"])
            query_text = row["question"]
            if query_id not in queries:
                queries[query_id] = {"id": query_id, "contents": query_text, "relevant": []}
                if include_negatives:
                    queries[query_id]["not_relevant"] = []
            query = queries[query_id]
            title = row["passage_title"].replace("\n", " ").replace("\r", " ").strip()
            text = row["passage_text"].replace("\n", " ").replace("\r", " ").strip()
            if len(title) > 0:
                text = text + " " + title
            if text not in passages:
                pid += 1
                passages[text] = {"id": f"p.{str(pid)}", "contents": text}
            relevant = row["relevant"]
            if include_negatives:
                query["relevant" if relevant else "not_relevant"].append(f"p.{str(pid)}")
            elif relevant:
                query["relevant"].append(f"p.{str(pid)}")
        all_queries = len(queries)
        queries = [val for key, val in queries.items() if len(val["relevant"]) > 0]
        dropped = all_queries - len(queries)
        logging.info("Dropped %d of %d queries due to the empty relevant list", dropped, all_queries)
        passages = [val for key, val in passages.items()]
        return queries, passages


class POLQATask(RetrievalTask):

    def __init__(self, splits=("test",)):
        super().__init__("polqa", "Other")
        self.splits = splits

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        logging.info("Preparing task %s", self.task_id)
        queries = self._read_queries()
        self.write_file(queries_path, queries)
        passages = self._read_passages()
        self.write_file(passages_path, passages)

    def _read_queries(self):
        queries = {}
        for split in self.splits:
            rows = load_dataset("ipipan/polqa", name="pairs", split=split)
            for row in rows:
                query_id = str(row["question_id"])
                query_text = row["question"]
                if query_id not in queries:
                    queries[query_id] = {"id": query_id, "contents": query_text, "relevant": []}
                query = queries[query_id]
                passage_id = row["passage_id"]
                relevant = row["relevant"]
                if relevant:
                    query["relevant"].append(passage_id)
        all_queries = len(queries)
        queries = [val for key, val in queries.items() if len(val["relevant"]) > 0]
        dropped = all_queries - len(queries)
        logging.info("Dropped %d of %d queries due to the empty relevant list", dropped, all_queries)
        return queries

    def _read_passages(self):
        passages = []
        rows = load_dataset("ipipan/polqa", name="passages", split="train")
        for row in rows:
            title = row["title"].replace("\n", " ").replace("\r", " ").strip()
            text = row["text"].replace("\n", " ").replace("\r", " ").strip()
            if len(title) > 0:
                text = text + " " + title
            passage_id = row["id"]
            passages.append({"id": passage_id, "contents": text})
        return passages


class GPTExamsTask(RetrievalTask):

    def __init__(self):
        super().__init__("gpt-exams", "Other")

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        logging.info("Preparing task %s", self.task_id)
        queries, passages = self._read_data()
        self.write_file(passages_path, passages)
        self.write_file(queries_path, queries)

    def _read_data(self):
        rows = load_dataset("sdadas/gpt-exams", split="train")
        queries, passages = [], []
        for row in rows:
            row_id = row["_id"]
            query = row["question"].replace("\n", " ").replace("\r", " ").strip()
            passage = row["answer"].replace("\n", " ").replace("\r", " ").strip()
            queries.append({"id": f"q.{row_id}", "contents": query, "relevant": [f"p.{row_id}"]})
            passages.append({"id": f"p.{row_id}", "contents": passage})
        return queries, passages


class LocalMSMarcoTask(RetrievalTask):

    def __init__(self, source_dir: str, split="eval", chatgpt=True):
        super().__init__("msmarco-pl", "Other")
        self.split = split
        self.source_dir = source_dir
        self.chatgpt = chatgpt

    def prepare_task(self, data_dir: str):
        if self.exists(data_dir): return
        passages_path = self.passages_path(data_dir)
        queries_path = self.queries_path(data_dir)
        logging.info("Preparing task %s", self.task_id)
        queries = self._read_queries()
        self.write_file(queries_path, queries)
        passages = self._read_passages()
        self.write_file(passages_path, passages)

    def _read_queries(self):
        file_name = f"queries.{self.split}{'.chatgpt' if self.chatgpt else ''}.jsonl"
        queries_path = os.path.join(self.source_dir, file_name)
        queries = {}
        with open(queries_path, "r", encoding="utf-8") as queries_file:
            for line in queries_file:
                value = json.loads(line.strip())
                query_id = value["id"]
                query = {"id": query_id, "contents": value["translation"], "relevant": set()}
                queries[query_id] = query
        triples_path = os.path.join(self.source_dir, f"qrels.{self.split}.tsv")
        with open(triples_path, "r", encoding="utf-8") as triples_file:
            for line in triples_file:
                ids = line.strip().split("\t")
                assert len(ids) == 4, ids
                query_id, pos_id, rel = ids[0], ids[2], ids[3]
                assert rel == "1"
                queries[query_id]["relevant"].add(pos_id)
        results = []
        for query in queries.values():
            relevant = query["relevant"]
            if len(relevant) == 0:
                continue
            query["relevant"] = list(relevant)
            results.append(query)
        return results

    def _read_passages(self):
        passages_path = os.path.join(self.source_dir, "collection.jsonl")
        passages = []
        with open(passages_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                value = json.loads(line.strip())
                passages.append({"id": value["id"], "contents": value["translation"]})
        return passages
