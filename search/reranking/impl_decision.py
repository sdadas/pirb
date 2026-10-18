"""Pointwise reranking through S1MB's typed decision adapters."""

import ast
import importlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

from s1mb.data import InferenceCase, Option, Question, check_probabilities

from search.reranking.base import RerankerBase


@lru_cache(maxsize=None)
def resolve_adapter(name):
    """Find a class without importing every optional S1MB backend."""
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
