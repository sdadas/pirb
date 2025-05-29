import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
from joblib import load, dump
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryRequest, PointStruct, QueryResponse, Distance, VectorParams
from backend.base import DenseBackend
from data import IndexResult


class QdrantBackend(DenseBackend):

    def __init__(self, index_dir: str, host: str, port: int):
        super().__init__(index_dir)
        self.index_ids_path = os.path.join(self.index_dir, "ids.bin")
        self.collection = index_dir.replace("/", "_").replace("\\", "_")
        self.host = host
        self.port = port
        self.bs = 1024
        self.client = QdrantClient(host=host, port=port)
        self._ids = None

    def build(self, embeddings, ids: List[str]):
        embeddings = np.vstack(embeddings)
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        vector_dim = embeddings.shape[1]
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE
            )
        )
        # noinspection PyTypeChecker
        requests = [PointStruct(id=i, vector=embeddings[i]) for i in range(len(ids))]
        for i in range(0, len(requests), self.bs):
            batch = requests[i:i + self.bs]
            self.client.upsert(self.collection, batch)
        dump(ids, self.index_ids_path)
        self._ids = ids

    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self._ids is None:
            self.open()
        vectors = embeddings.tolist()
        requests: List[QueryRequest] = [QueryRequest(query=vectors[i], limit=top_k) for i in range(len(qids))]
        responses: List[QueryResponse] = self.client.query_batch_points(self.collection, requests)
        results = defaultdict(list)
        for i, qid in enumerate(qids):
            response = responses[i]
            results[qid] = [IndexResult(self._ids[point.id], point.score) for point in response.points]
        return results

    def open(self):
        self._ids = load(self.index_ids_path)

    def exists(self) -> bool:
        return os.path.exists(self.index_ids_path) and self.client.collection_exists(self.collection)

    def close(self):
        self.client.close()
