import copy
import logging
from typing import Any
from .base import *
from data import RetrievalTask


class AutoIndex:

    @staticmethod
    def from_config(config: Dict, task: RetrievalTask, args):
        # noinspection PyTypeChecker
        cache_dir: str = os.path.join(args.cache_dir, task.task_cache_name())
        index_name = config["name"]
        index_type = config.get("type", None)
        use_bt = args.use_bettertransformer if hasattr(args, "use_bettertransformer") else False
        tasks_config = config.get("tasks", None)
        if tasks_config is not None and isinstance(tasks_config, dict):
            task_overrides = AutoIndex.find_task_overrides(task.task_id, tasks_config)
            if task_overrides is not None and isinstance(task_overrides, dict):
                logging.info("Overriding default params for task {}".format(task.task_id))
                config = copy.deepcopy(config)
                for key, val in task_overrides.items():
                    if "." in key:
                        AutoIndex.set_nested_value(config, key, val)
                    else:
                        config[key] = val
        if index_type == "hybrid":
            from .hybrid import HybridIndex
            child_configs = config["models"]
            indices = [AutoIndex.from_config(child_config, task, args) for child_config in child_configs]
            k0 = config.get("k0", 100)
            rerank_limit = args.rerank_limit if hasattr(args, "rerank_limit") else None
            strategy = config.get("strategy", None)
            return HybridIndex(args.data_dir, index_name, indices, k0, strategy, task, rerank_limit, use_bt)
        elif index_type == "splade":
            from .splade import SpladeIndex
            return SpladeIndex(config, data_dir=cache_dir, use_bettertransformer=use_bt)
        elif index_name in ("sparse", "bm25"):
            from .lucene import LuceneIndex
            lang = config.get("lang", None)
            if hasattr(task, "lang") and getattr(task, "lang") is not None:
                lang = getattr(task, "lang")
            if lang is None:
                lang = "pl"
            threads = config.get("threads", 8)
            return LuceneIndex(cache_dir, lang, threads)
        elif index_type == "late_interaction":
            from .late_interaction import LateInteractionIndex
            return LateInteractionIndex(data_dir=cache_dir, encoder=config)
        else:
            from .dense import DenseIndex
            return DenseIndex(data_dir=cache_dir, encoder=config, use_bettertransformer=use_bt)

    @staticmethod
    def set_nested_value(d: Dict, key: str, value: Any):
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @staticmethod
    def find_task_overrides(task_id: str, tasks_config: Dict):
        task_overrides = tasks_config.get(task_id, None)
        if task_overrides is None:
            for key, val in tasks_config.items():
                if key.startswith("~"):
                    task_substring = key[1:].lower()
                    if task_substring in task_id.lower():
                        task_overrides = val
        return task_overrides
