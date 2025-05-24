from abc import ABC, abstractmethod
from typing import List, Dict
from data import IndexResult


class DenseBackend(ABC):

    def __init__(self, index_dir: str):
        self.index_dir = index_dir

    @abstractmethod
    def build(self, embeddings, ids: List[str]):
        raise NotImplementedError()

    @abstractmethod
    def search(self, embeddings, qids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        raise NotImplementedError()

    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError()

    def close(self):
        pass

    def supports_tensors(self):
        return False
