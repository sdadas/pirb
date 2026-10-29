"""Performance measurements for a single dense benchmark run."""

from contextlib import contextmanager
from time import perf_counter

import torch


class BenchmarkStats:
    def __init__(self):
        self.values = dict(documents=0, queries=0, indexing_seconds=0.0,
                           search_seconds=0.0, query_encoding_seconds=0.0,
                           max_gpu_memory_reserved_bytes=0)
        self.devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.synchronize()
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    @contextmanager
    def measure(self, key):
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.values[key] += perf_counter() - start

    def snapshot(self):
        # Sum of per-device peaks, not necessarily simultaneous. Only memory
        # reserved by PyTorch in this process is visible (not remote workers).
        self.values["max_gpu_memory_reserved_bytes"] = sum(
            torch.cuda.max_memory_reserved(device) for device in self.devices
        )
        return summarize_stats([self.values])


def summarize_stats(records):
    totals = {key: sum(record[key] for record in records) for key in (
        "documents", "queries", "indexing_seconds", "search_seconds", "query_encoding_seconds"
    )}
    totals["max_gpu_memory_reserved_bytes"] = max(
        (record["max_gpu_memory_reserved_bytes"] for record in records), default=0
    )
    dimensions = sorted({record["embedding_dim"] for record in records
                         if record.get("embedding_dim") is not None})
    totals["embedding_dim"] = dimensions[0] if len(dimensions) == 1 else (dimensions or None)
    totals["search_without_encoding_seconds"] = max(
        0.0, totals["search_seconds"] - totals["query_encoding_seconds"]
    )
    for rate, count, duration in (
        ("documents_per_second", "documents", "indexing_seconds"),
        ("queries_per_second", "queries", "search_seconds"),
        ("queries_per_second_without_encoding", "queries", "search_without_encoding_seconds"),
    ):
        totals[rate] = totals[count] / totals[duration] if totals[duration] > 0 else None
    for latency, duration in (
        ("mean_query_latency_seconds", "search_seconds"),
        ("mean_query_latency_without_encoding_seconds", "search_without_encoding_seconds"),
    ):
        totals[latency] = totals[duration] / totals["queries"] if totals["queries"] > 0 else None
    return totals
