"""Pointwise reranking through S1MB's typed decision adapters."""

import ast
import importlib
import inspect
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from data import IndexResult

from s1mb.data import InferenceCase, Option, Question, check_probabilities
from s1mb.adapters.typesafe import TypeSafeAdapter

from search.reranking.base import RerankerBase


class OpenRouterAdapter(TypeSafeAdapter):
    """Use S1MB's TypeSafe protocol through OpenRouter's System One API."""

    provider = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai"
    endpoint = "/api/v1/systemone"

    def __init__(self, model: str, case_batch_size: int = 4):
        if isinstance(case_batch_size, bool) or not isinstance(case_batch_size, int) or not 1 <= case_batch_size <= 32:
            raise ValueError("case_batch_size must be an integer within 1..32")
        self.case_batch_size = case_batch_size
        super().__init__(model)

    def predict_batch(self, cases):
        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(self.predict, cases))


@lru_cache(maxsize=None)
def resolve_adapter(name):
    """Find a class without importing every optional S1MB backend."""
    if name == "OpenRouterAdapter":
        return OpenRouterAdapter
    import s1mb.adapters

    matches = []
    for root in s1mb.adapters.__path__:
        for path in Path(root).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body):
                matches.append(f"s1mb.adapters.{path.stem}.{name}")
    if name.startswith("s1mb.adapters."):
        matches = [name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous S1MB adapter {name!r}: {matches}. "
                         "Use a fully qualified s1mb.adapters.module.Class name.")
    module, class_name = matches[0].rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)


def render(value, query, passage):
    if isinstance(value, str):
        return value.format(query=query, passage=passage)
    if isinstance(value, dict):
        return {key: render(item, query, passage) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, query, passage) for item in value]
    return value


class DecisionReranker(RerankerBase):
    def __init__(self, **config):
        self.save_scores = config.get("save_scores", True)
        if not isinstance(self.save_scores, bool):
            raise ValueError("save_scores must be a boolean")
        self._score_cache = {}
        self._score_log = None
        self.template = config["decision_template"]
        self.task = self.template["type"]
        self.batch_size = config.get("batch_size", 32)
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.task not in ("noul", "score"):
            raise ValueError("decision_template.type must be noul or score")
        # Validate templates and criteria before loading expensive model weights.
        self._case("", "", 0)
        adapter_class = resolve_adapter(config["model_adapter"])
        model_kwargs = dict(config.get("model_kwargs", {}))
        signature = inspect.signature(adapter_class)
        model_parameter = next((key for key in ("model", "checkpoint", "model_name", "model_id")
                                if key in signature.parameters), None)
        if model_parameter is None:
            raise ValueError(f"Cannot identify model argument for {adapter_class.__name__}")
        model_kwargs.setdefault(model_parameter, config["reranker_name"])
        try:
            signature.bind(**model_kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid model_kwargs for {adapter_class.__name__}: {exc}") from exc
        self.adapter = adapter_class(**model_kwargs)

    def configure_score_log(self, name):
        """Attach benchmark identity; plain text-only calls have no dataset IDs."""
        if not self.save_scores:
            return
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")
        if not safe_name or safe_name in (".", ".."):
            raise ValueError("Invalid configuration name for score log")
        path = Path("_logs") / f"{safe_name}.jsonl"
        if self._score_log == path:
            return
        self._score_cache = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        # A killed append may leave an incomplete final line. Only that tail is discarded.
        with path.open("a+b") as stream:
            stream.seek(0)
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                try:
                    row = json.loads(line)
                    docs, scores = row["docs"], row["scores"]
                    if (not isinstance(docs, list) or not isinstance(scores, list)
                            or len(docs) != len(scores) or len(set(docs)) != len(docs)
                            or not all(isinstance(d, str) for d in docs)
                            or not all(isinstance(s, (int, float)) and not isinstance(s, bool)
                                       and math.isfinite(s) for s in scores)
                            or not isinstance(row["task_id"], str) or not isinstance(row["id"], str)):
                        raise ValueError("Invalid score record")
                    self._score_cache[row["task_id"], row["id"]] = [IndexResult(d, s) for d, s in zip(docs, scores)]
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid score log {path} at byte {offset}") from exc
        self._score_log = path

    def load_scores(self, task_id, query_id):
        """Return a completed query, or None when inference is needed."""
        return self._score_cache.get((task_id, query_id)) if self.save_scores else None

    def store_scores(self, task_id, query_id, results):
        """Append the final top-k ranking only after the whole query succeeds."""
        if not self.save_scores:
            return
        if self._score_log is None:
            raise ValueError("Call configure_score_log with the configuration name first")
        row = {"task_id": task_id, "id": query_id,
               "docs": [r.id for r in results], "scores": [r.score for r in results]}
        with self._score_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._score_cache[task_id, query_id] = list(results)

    def _case(self, query, passage, index):
        template = render(self.template, query, passage)
        criteria = template["criteria"]
        if self.task == "noul":
            if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                raise ValueError('Noul criteria must be an object with "false" and "true" descriptions')
            options = [Option(id=key, description=value) for key, value in criteria.items()]
        else:
            if not isinstance(criteria, list):
                raise ValueError("Score criteria must be a list")
            options = [
                Option(id=str(i), description=item, value=i) if isinstance(item, str)
                else Option(id=str(i), description=item["description"], value=item["value"])
                for i, item in enumerate(criteria)
            ]
        state = template["state"]
        return InferenceCase(
            case_id=str(index), state=state, state_json=json.dumps(state, ensure_ascii=False),
            questions=[Question(id="relevance", task=self.task,
                                instructions=template["instructions"],
                                system_prompt=template.get("system_prompt", ""), options=options)],
        )

    def rerank_pairs(self, queries, docs, proba=False):
        """Return P(true) or expected numeric score; proba does not rescale scores."""
        if len(queries) != len(docs):
            raise ValueError("queries and docs must have equal length")
        if self.adapter is None:
            raise RuntimeError("Decision reranker is closed")
        scores = []
        batch_predict = getattr(self.adapter, "predict_batch", None)
        for start in range(0, len(docs), self.batch_size):
            cases = [self._case(queries[i], docs[i], i)
                     for i in range(start, min(start + self.batch_size, len(docs)))]
            outputs = batch_predict(cases) if callable(batch_predict) else [self.adapter.predict(c) for c in cases]
            if len(outputs) != len(cases):
                raise ValueError("S1MB returned an incorrect number of cases")
            for case, predictions in zip(cases, outputs, strict=True):
                if isinstance(predictions, Exception):
                    raise RuntimeError(f"S1MB failed for case {case.case_id}") from predictions
                if len(predictions) != 1:
                    raise ValueError("S1MB must return one prediction per query-document pair")
                prediction = predictions[0]
                question = case.questions[0]
                if prediction.case_id != case.case_id or prediction.question_id != question.id:
                    raise ValueError("S1MB prediction IDs do not match the request")
                if prediction.error is not None:
                    raise RuntimeError(f"S1MB inference failed: {prediction.error}")
                probabilities = prediction.probabilities
                if probabilities is None:
                    raise ValueError("S1MB returned no probabilities")
                check_probabilities(probabilities, [o.id for o in question.options])
                score = (probabilities["true"] if self.task == "noul" else
                         sum(probabilities[o.id] * o.value for o in question.options))
                scores.append(float(score))
        return scores

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
