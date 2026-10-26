import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from threading import Lock
from openai import OpenAI, omit
from search.reranking.base import RerankerBase
from typing import Dict, List


class DecisionReranker(RerankerBase):

    def __init__(self, reranker_name, api_base: str | List[str], decision_template: Dict, api_key=None,
                 batch_size=32, timeout=180, max_retries=3, **kwargs):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if decision_template["type"] not in ("noul", "score"):
            raise ValueError("decision_template.type must be noul or score")
        urls = [api_base] if isinstance(api_base, str) else api_base
        if not isinstance(urls, list) or not urls or any(not isinstance(url, str) or not url.strip() for url in urls):
            raise ValueError("api_base must be a non-empty URL or list of URLs")
        if api_key and api_key.startswith("$"):
            name = api_key[1:]
            api_key = os.environ.get(name)
            if not api_key:
                raise ValueError(f"Environment variable {name!r} is missing or empty")
        self.model = reranker_name
        self.template = decision_template
        self.batch_size = batch_size
        self.headers = {} if api_key else {"Authorization": omit}
        self.clients = []
        try:
            for url in urls:
                self.clients.append(OpenAI(
                    base_url=url, api_key=api_key or "unused", timeout=timeout,
                    max_retries=max_retries,
                ))
        except Exception:
            self.close()
            raise
        self._clients = cycle(self.clients)
        self._client_lock = Lock()

    def _score(self, query, passage):
        def render(value):
            if isinstance(value, str):
                return value.format(query=query, passage=passage)
            if isinstance(value, list):
                return [render(item) for item in value]
            if isinstance(value, dict):
                return {key: render(item) for key, item in value.items()}
            return value

        template = render(self.template)
        question = {key: template[key] for key in ("type", "instructions", "criteria")}
        if template.get("system_prompt"):
            question["instructions"] = template["system_prompt"] + "\n\n" + question["instructions"]
        with self._client_lock:
            client = next(self._clients)
        response = client.post("/systemone", cast_to=dict, options={"headers": self.headers}, body={
            "model": self.model, "state": template["state"],
            "questions": {"relevance": question},
        })
        answer = response["answers"]["relevance"]
        if answer["type"] != question["type"]:
            raise ValueError("API returned a different decision type")
        score = float(answer[question["type"]])
        if not math.isfinite(score) or (question["type"] == "noul" and not 0 <= score <= 1):
            raise ValueError("API returned an invalid score")
        return score

    def rerank_pairs(self, queries, docs, proba=False):
        """Return native score or P(true), preserving input order."""
        if len(queries) != len(docs):
            raise ValueError("queries and docs must have equal length")
        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            return list(pool.map(self._score, queries, docs))

    def close(self):
        for client in self.clients:
            client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
