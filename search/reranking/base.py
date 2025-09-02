from abc import ABC, abstractmethod
from typing import List


class RerankerBase(ABC):

    @abstractmethod
    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        raise NotImplementedError()

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        return self.rerank_pairs([query] * len(docs), docs, proba)
