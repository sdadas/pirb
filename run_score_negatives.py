import json
import logging
import os.path
import random

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from utils.system import set_java_env
set_java_env()
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSeq2SeqLM
from data import RawJsonlTask, IndexInput
from search import SearchIndex, AutoIndex, RerankerHybrid


FLAN_PRP_PROMPT = '''Question: Given a query "{0}", which of the following two passages is more relevant to the query?

passage A: {1}

passage B: {2}

Output the identifier of the more relevant passage. The answer must be passage A or passage B.
Answer:'''


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
        self.indices: List[SearchIndex] = self.create_indices()
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
        if self.args.use_bettertransformer:
            conf["use_bettertransformer"] = True
        self.reraker = RerankerHybrid(**conf)

    def build(self):
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
            results = index.search(queries, max(self.args.max_negs_per_index, 32))
            index_results.append(results)
        self.create_reranker()
        output_path = os.path.join(self.args.data_dir, self.task.task_id, self.args.output_name)
        with open(output_path, "w", encoding="utf-8") as output_file:
            for query in tqdm(queries, desc="Scoring docs for queries"):
                out = self.process_query(query, passages, index_results)
                output_file.write(json.dumps(out, ensure_ascii=False))
                output_file.write("\n")
                output_file.flush()

    def process_query(self, query: IndexInput, passages: Dict, index_results: List[Dict]):
        negs = {}
        out = {"query_id": query.id, "positive": query.relevant, "negative": negs}
        ids2docs = lambda ids: [passages.get(val) for val in ids]
        unique_docs = set(query.relevant)
        out["positive_scores"] = self.reraker.rerank(query.text, ids2docs(query.relevant), proba=True)
        for idx, result in enumerate(index_results):
            hits = result.get(query.id)
            if hits is None:
                hits = []
            docids = [hit.id for hit in hits if hit.id not in unique_docs]
            if len(docids) > self.args.max_negs_per_index:
                docids = docids[:self.args.max_negs_per_index]
            key = self.retriever_keys[idx]
            unique_docs.update(docids)
            scores = self.reraker.rerank(query.text, ids2docs(docids), proba=True) if len(docids) > 0 else []
            negs[key] = {"docs": docids, "scores": scores}
        return out


class PermutationNegsBuilder(HardNegsBuilder):

    def __init__(self, args: HardNegsArgs):
        super().__init__(args)
        self.tokenizer = None
        self.model = None
        self.prp_token_passage = 5454
        self.prp_a = 71
        self.prp_b = 272
        self.prp_prompt = FLAN_PRP_PROMPT
        self.prp_max_text_len = 100
        self.prp_max_passages = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_reranker(self):
        model_name = self.args.reranker.removeprefix("prp:")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()

    def process_query(self, query: IndexInput, passages: Dict, index_results: List[Dict]):
        out = {"query_id": query.id, "positive": query.relevant}
        if len(query.relevant) > 1:
            scores, order = self._rerank_sort(query, passages, query.relevant)
            out["positive_scores"] = scores
            out["positive_order"] = order
        unique_docs = set(query.relevant)
        negs = []
        for idx, result in enumerate(index_results):
            hits = result.get(query.id)
            if hits is None:
                hits = []
            docids = [hit.id for hit in hits if hit.id not in unique_docs]
            if len(docids) > self.args.max_negs_per_index:
                docids = docids[:self.args.max_negs_per_index]
            unique_docs.update(docids)
            negs.extend([(docid, idx) for idx, docid in enumerate(docids)])
        random.shuffle(negs)
        negs = negs[:self.prp_max_passages]
        negs.sort(key=lambda v: v[1])
        negs = [val[0] for val in negs]
        out["negative"] = negs
        scores, order = self._rerank_sort(query, passages, negs)
        out["negative_scores"] = scores
        out["negative_order"] = order
        return out

    def _rerank_sort(self, query: IndexInput, passages: Dict, ids: List[str]):
        docs = [passages.get(val) for val in ids]
        all_pairs = self._create_all_pairs(query, docs)
        i = 0
        all_score = [0 for _ in range(len(docs))]
        while i < len(all_pairs):
            batch = all_pairs[i: i + 10]
            i += 10
            texts = [psg[0] for psg in batch]
            features = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            features['decoder_input_ids'] = torch.tensor([[0, self.prp_token_passage]] * len(batch)).long()
            features = {k: v.to(self.device) for k, v in features.items()}
            with torch.no_grad():
                scores = self.model(**features).logits[:, -1]
            for score, psg in zip(scores, batch):
                if score[self.prp_a] > score[self.prp_b]:
                    all_score[psg[1]] += 1
                elif score[self.prp_b] > score[self.prp_a]:
                    all_score[psg[2]] += 1
                else:
                    all_score[psg[1]] += 0.5
                    all_score[psg[2]] += 0.5
        all_score = [s + 1 / (10 + r) for r, s in enumerate(all_score)]
        return all_score, np.argsort(all_score)[::-1].tolist()

    def _create_all_pairs(self, query: IndexInput, docs: List[str]):
        q = self.abbreviate(query.text)
        all_pairs = []
        for i in range(len(docs)):
            for j in range(len(docs)):
                if i == j: continue
                doc1 = self.abbreviate(docs[i])
                doc2 = self.abbreviate(docs[j])
                prompt = self.prp_prompt.format(q, doc1, doc2)
                all_pairs.append([prompt, i, j])
        return all_pairs

    def abbreviate(self, v: str):
        return " ".join(v.split()[:self.prp_max_text_len])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([HardNegsArgs])
    args = parser.parse_args_into_dataclasses()[0]
    if args.reranker.startswith("prp:"):
        builder = PermutationNegsBuilder(args)
    else:
        builder = HardNegsBuilder(args)
    builder.build()
