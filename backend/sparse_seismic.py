import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from backend.base import SparseBackend
from data import IndexResult


class SparseSeismicBackend(SparseBackend):

    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        try:
            from seismic import SeismicIndexLV, get_seismic_string
        except ImportError as exc:
            raise ImportError("The seismic backend requires the 'seismic' extra (pyseismic-lsr)") from exc
        self._index_type = SeismicIndexLV
        self._string_dtype = get_seismic_string()
        self.path = Path(index_dir) / "vectors.index.seismic"
        self._index = None

    def build(self, batches):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path.parent) as tmp_dir:
            index_base = Path(tmp_dir) / "vectors"
            index_path = Path(tmp_dir) / self.path.name
            vectors_path = Path(tmp_dir) / "vectors.jsonl"
            with vectors_path.open("w", encoding="utf-8") as output:
                for embeddings, ids in batches:
                    if np.any(embeddings.data < 0):
                        raise ValueError("Seismic requires nonnegative sparse weights")
                    for doc_id, start, end in zip(ids, embeddings.indptr[:-1], embeddings.indptr[1:]):
                        if start != end:
                            dimensions = embeddings.indices[start:end]
                            weights = embeddings.data[start:end]
                            vector = {str(i): float(weight) for i, weight in zip(dimensions, weights)}
                            output.write(json.dumps({"id": doc_id, "vector": vector}) + "\n")
            index = None
            if vectors_path.stat().st_size:
                index = self._index_type.build(str(vectors_path), load_content=False)
                index.save(str(index_base))
            else:
                index_path.touch()
            os.replace(index_path, self.path)
        self._index = index

    def search(self, embeddings, qids, top_k):
        results = {qid: [] for qid in qids}
        if not qids or top_k <= 0:
            return results
        if self._index is None:
            if self.path.stat().st_size == 0:
                return results
            self._index = self._index_type.load(str(self.path))
        if np.any(embeddings.data < 0):
            raise ValueError("Seismic requires nonnegative sparse weights")

        query_positions, components, weights = [], [], []
        for pos, (start, end) in enumerate(zip(embeddings.indptr[:-1], embeddings.indptr[1:])):
            if start != end:
                query_positions.append(str(pos))
                components.append(embeddings.indices[start:end].astype(self._string_dtype))
                weights.append(embeddings.data[start:end])
        if query_positions:
            matches = self._index.batch_search(
                queries_ids=np.array(query_positions, dtype=self._string_dtype),
                query_components=components,
                query_values=weights,
                k=min(top_k, self._index.len),
                query_cut=max(map(len, components)),
                heap_factor=0.7,
            )
            for hits in matches:
                for position, score, doc_id in hits:
                    results[qids[int(position)]].append(IndexResult(doc_id, float(score)))
        return results

    def exists(self):
        return self.path.is_file()

    def close(self):
        self._index = None
