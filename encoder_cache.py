"""Run-scoped cache of loaded encoders, independent of dataset indexes."""

import json


class EncoderCache:
    def __init__(self):
        self._entries = {}

    def get_or_load(self, config, loader):
        # These options affect retrieval, not the loaded encoder. In particular,
        # backend configs can contain a dataset-specific index directory.
        excluded = {"backend", "tasks", "q_prefix", "p_prefix", "q_prefix_name",
                    "p_prefix_name", "normalize"}
        key = json.dumps({k: v for k, v in config.items() if k not in excluded}, sort_keys=True)
        slot = config["name"]
        entry = self._entries.get(slot)
        if entry is not None and entry[0] == key:
            return entry[1]
        # Do not retain all GPU copies when task overrides change a model.
        self._entries.pop(slot, None)
        del entry
        model = loader()
        self._entries[slot] = (key, model)
        return model

    def clear(self):
        self._entries.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()
