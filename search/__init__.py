from .base import *
from .dense import *
from .lucene import *
from .splade import *
from .hybrid import *


class AutoIndex:

    @staticmethod
    def from_config(config: Dict, task: RetrievalTask, args):
        cache_dir = os.path.join(args.cache_dir, task.task_cache_name())
        index_name = config["name"]
        index_type = config.get("type", None)
        use_bt = args.use_bettertransformer if hasattr(args, "use_bettertransformer") else False
        raw_mode = args.raw_mode if hasattr(args, "raw_mode") else True
        tasks_config = config.get("tasks", None)
        if tasks_config is not None and isinstance(tasks_config, dict):
            task_overrides = tasks_config.get(task.task_id, None)
            if task_overrides is not None and isinstance(task_overrides, dict):
                config = dict(config)
                for key, val in task_overrides.items():
                    config[key] = val
        if index_type == "hybrid":
            child_configs = config["models"]
            indices = [AutoIndex.from_config(child_config, task, args) for child_config in child_configs]
            k0 = config.get("k0", 100)
            rerank_limit = args.rerank_limit
            strategy = config.get("strategy", None)
            return HybridIndex(args.data_dir, index_name, indices, k0, strategy, task, rerank_limit, use_bt)
        elif index_type == "splade":
            return SpladeIndex(config, data_dir=cache_dir, use_bettertransformer=use_bt, threads=args.threads)
        elif index_name in ("sparse", "bm25"):
            lang = config.get("lang", "pl")
            return LuceneIndex(cache_dir, lang, args.threads)
        else:
            return DenseIndex(data_dir=cache_dir, encoder=config, use_bettertransformer=use_bt, raw_mode=raw_mode)
