import os
import asyncio
from collections import Counter
from dataclasses import dataclass
from random import choice
from typing import List, Iterable, Tuple, Dict, Callable
from search.reranking import RerankerBase


@dataclass
class LLMCall:
    messages: List
    response: str = None


class LLMReranker(RerankerBase):

    def __init__(self, **config):
        self.batch_size = config.get("batch_size", 1)
        self.name = config["reranker_name"]
        self.api_name = config.get("api_name", self.name)
        self.api_base = config["api_base"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", None)
        self.extra_body = config.get("extra_body", {})
        self.api_key = config.get("api_key", "-")
        if self.api_key and self.api_key.startswith("$"):
            self.api_key = os.environ[self.api_key[1:]]
        self.error_on_max_len = False

    def call_many_async(self, batch: List[LLMCall]) -> Iterable[LLMCall]:
        return asyncio.run(self._call_many_async(batch))

    async def _call_many_async(self, batch: List[LLMCall]):
        semaphore = asyncio.Semaphore(self.batch_size)
        responses = await asyncio.gather(*[self.call_async(req, semaphore) for req in batch])
        return responses

    async def call_async(self, request: LLMCall, semaphore: asyncio.Semaphore) -> LLMCall:
        from openai import AsyncOpenAI
        async with semaphore:
            api_base = self.api_base
            if isinstance(self.api_base, list):
                api_base = choice(self.api_base)
            client = AsyncOpenAI(api_key=self.api_key, base_url=api_base)
            resp = await client.chat.completions.create(
                messages=request.messages,
                model=self.api_name,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body
            )
            completion = resp.choices[0]
            assert completion.finish_reason != "length" or not self.error_on_max_len, "model max length reached"
            request.response = completion.message.content
            await client.close()
            return request

    def call(self, request: LLMCall):
        from openai import OpenAI
        api_base = self.api_base
        if isinstance(self.api_base, list):
            api_base = choice(self.api_base)
        client = OpenAI(api_key=self.api_key, base_url=api_base)
        resp = client.chat.completions.create(
            messages=request.messages,
            model=self.api_name,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body=self.extra_body
        )
        completion = resp.choices[0]
        assert completion.finish_reason != "length" or not self.error_on_max_len, "model max length reached"
        request.response = completion.message.content
        client.close()
        return request

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        raise NotImplementedError("Pair based reranking not possible for LLMReranker")

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        return NotImplementedError()


class RankGPTReranker(LLMReranker):
    """
    Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
    https://aclanthology.org/2023.emnlp-main.923/
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.sliding_window_size = config.get("sliding_window_size", 20)
        self.sliding_window_overlap = self.sliding_window_size // 2
        self._init_prompts()

    def _init_prompts(self):
        self.system_prompt = (
            "Jesteś inteligentnym asystentem, który potrafi sortować dokumenty według ich relewantności do zapytania."
        )
        self.first_prompt = (
            "Podam {num_docs} dokumentów, każdy oznaczony unikalnym identyfikatorem w nawiasach kwadratowych [].\n"
            "Posortuj te dokumenty według ich relewantności do zapytania: {query}."
        )
        self.assistant_response = (
            "Zrozumiałem, oczekuję na dokumenty."
        )
        self.assistant_ack = (
            "Otrzymałem dokument [{num}]."
        )
        self.last_prompt = (
            "Posortuj powyższe dokumenty zaczynając od tych, które są najlepiej dopasowane do pytania i w najlepszy "
            "sposób odpowiadają na nie. Odpowiedź powinna zawierać listę identyfikatorów posortowaną według "
            "relewantności. Najbardziej relewantny dokument powinien być na pierwszej pozycji. Format odpowiedzi to "
            "[] > [] > [] np. [1] > [2] > [3]. Odpowiedz tylko listą identyfikatorów, bez dodatkowych komentarzy czy "
            "wyjaśnień.\n\nPytanie: {query}"
        )

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        docs_copy = docs.copy()
        slice_start = len(docs) - self.sliding_window_size
        while slice_start >= 0:
            slice_end = slice_start + self.sliding_window_size
            docs_slice = docs_copy[slice_start:slice_end]
            sorted_slice = self._rerank_slice(query, docs_slice)
            for idx, slice_idx in enumerate(range(slice_start, slice_end)):
                doc = sorted_slice[idx]
                docs_copy[slice_idx] = doc
            slice_start -= self.sliding_window_overlap
        docs_scores = {doc: float(1.0 / (idx + 1.0)) for idx, doc in enumerate(docs_copy)}
        scores = [docs_scores[doc] for doc in docs]
        return scores

    def _rerank_slice(self, query: str, docs: List[str]):
        messages = self._get_prompt(query, docs)
        request = LLMCall(messages)
        self.call(request)
        response = "".join([(c if c.isdigit() else " ") for c in request.response]).strip()
        ranks = self._create_ranks(response, docs)
        sorted_docs = [docs[i] for i in ranks]
        return sorted_docs

    def _create_ranks(self, response: str, docs: List[str]):
        ranks = [int(x) - 1 for x in response.split()]
        ranks_ids = set()
        possible_ranks = set(range(len(docs)))
        result = []
        for rank in ranks:
            if rank not in ranks_ids and rank in possible_ranks:
                result.append(rank)
                ranks_ids.add(rank)
        for i in range(len(docs)):
            if i not in ranks_ids:
                result.append(i)
                ranks_ids.add(i)
        assert len(result) == len(docs)
        return result

    def _get_prompt(self, query: str, docs: List[str]):
        res = []
        res.append({"role": "system", "content": self.system_prompt})
        res.append({"role": "user", "content": self.first_prompt.format(num_docs=len(docs), query=query)})
        res.append({"role": "assistant", "content": self.assistant_response})
        for idx, doc in enumerate(docs):
            res.append({"role": "user", "content": f"[{idx + 1}] {doc}"})
            res.append({"role": "assistant", "content": self.assistant_ack.format(num=idx + 1)})
        res.append({"role": "user", "content": self.last_prompt.format(query=query)})
        return res


class PairwiseLLMReranker(LLMReranker):
    """
    Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting
    https://aclanthology.org/2024.findings-naacl.97/
    """

    def __init__(self, **config):
        super().__init__(**config)
        self._init_prompts()
        self.algorithm = config.get("algorithm", "allpair")
        algorithms = {"allpair": self.rerank_allpair, "sliding": self.rerank_sliding}
        self.algorithm_func: Callable = algorithms.get(self.algorithm)

    def _init_prompts(self):
        self.prompt = (
            "Poniżej podane zostaną dwa dokumenty oraz pytanie. Twoim zadaniem będzie odpowiedź, który z dokumentów "
            "jest bardziej relewanty do zadanego pytania, czyli który zawiera więcej informacji pozwalających "
            "odpowiedzieć na to pytanie.\n\n"
            "# Dokument 1:\n{doc1}\n\n"
            "# Dokument 2:\n{doc2}\n\n"
            "# Pytanie\n{query}\n\n"
            "Podaj numer dokumentu, który jest bardziej relewantny do zadanego pytania. Zwróć pojedynczą cyfrę jako "
            "odpowiedź: 1 lub 2. Nie dodawaj żadnych komentarzy ani wyjaśnień."
        )

    def _pairwise_scores(self, query: str, docs: List[str], pairs: List[Tuple], score_cache: Dict):
        calls = self._create_prompts(query, docs, pairs, score_cache)
        if len(calls) == 1:
            self.call(calls[0])
        else:
            _ = [val for val in self.call_many_async(calls)]
        for call in calls:
            try:
                res = int(call.response.strip())
            except ValueError:
                res = 0
            score_cache[call.text_pair] = res
        return [score_cache[(docs[pair[0]], docs[pair[1]])] for pair in pairs]

    def _create_prompts(self, query: str, docs: List[str], pairs: List[Tuple], score_cache: Dict):
        calls: List[LLMCall] = []
        for pair in pairs:
            text_pair = (docs[pair[0]], docs[pair[1]])
            if text_pair in score_cache:
                continue
            msg = self.prompt.format(query=query, doc1=docs[pair[0]], doc2=docs[pair[1]])
            call = LLMCall([{"role": "user", "content": msg}])
            call.text_pair = text_pair
            calls.append(call)
        return calls

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        return self.algorithm_func(query, docs)

    def rerank_allpair(self, query: str, docs: List[str]):
        all_pairs = []
        for i in range(len(docs)):
            for j in range(len(docs)):
                if i == j: continue
                all_pairs.append((i, j))
        scores = self._pairwise_scores(query, docs, all_pairs, {})
        doc_scores = Counter()
        for pair, score in zip(all_pairs, scores):
            doc1, doc2 = pair[0], pair[1]
            assert score == 0 or score == 1 or score == 2, "incorrect score"
            if score == 0:
                doc_scores[doc1] += 0.5
                doc_scores[doc2] += 0.5
            elif score == 1:
                doc_scores[doc1] += 1
            elif score == 2:
                doc_scores[doc2] += 1
        # handle ties
        for idx in range(len(docs)):
            doc_scores[docs[idx]] += 1 / (10 + idx)
        results = [float(doc_scores[idx]) for idx in range(len(docs))]
        return results

    def rerank_sliding(self, query: str, docs: List[str]):
        iters = max(0, min(10, len(docs)))
        docs_copy = docs.copy()
        score_cache = {}
        for i in range(iters):
            for j in range(len(docs_copy) - 1, i, -1):
                pair = (j - 1, j)
                score = self._pairwise_scores(query, docs_copy, [pair], score_cache)[0]
                if score > 1:
                    tmp = docs_copy[j]
                    docs_copy[j] = docs_copy[j - 1]
                    docs_copy[j - 1] = tmp
        docs_scores = {doc: float(1.0 / (idx + 1.0)) for idx, doc in enumerate(docs_copy)}
        scores = [docs_scores[doc] for doc in docs]
        return scores
