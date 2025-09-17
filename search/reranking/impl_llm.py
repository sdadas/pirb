import os
import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from random import choice
from threading import Lock
from typing import List, Iterable, Tuple, Dict, Callable, Any
from search.reranking import RerankerBase


@dataclass
class LLMCall:
    messages: List
    response: str = None


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    length_errors: int = 0


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
        self.usage = LLMUsage()
        self._usage_lock = Lock()

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
            self._report_usage(resp)
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
        self._report_usage(resp)
        completion = resp.choices[0]
        assert completion.finish_reason != "length" or not self.error_on_max_len, "model max length reached"
        request.response = completion.message.content
        client.close()
        return request

    def rerank_pairs(self, queries: List[str], docs: List[str], proba: bool = False):
        raise NotImplementedError("Pair based reranking not possible for LLMReranker")

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        return NotImplementedError()

    def _report_usage(self, response):
        with self._usage_lock:
            self.usage.calls += 1
            if response.usage:
                self.usage.input_tokens += response.usage.prompt_tokens
                self.usage.output_tokens += response.usage.completion_tokens
            if response.choices and len(response.choices) > 0:
                completion = response.choices[0]
                self.usage.length_errors += (1 if completion.finish_reason == "length" else 0)


class RankGPTReranker(LLMReranker):
    """
    Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
    https://aclanthology.org/2023.emnlp-main.923/

    APEER: Automatic Prompt Engineering Enhances Large Language Model Reranking
    https://dl.acm.org/doi/10.1145/3701716.3717574
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.sliding_window_size = config.get("sliding_window_size", 20)
        self.sliding_window_overlap = self.sliding_window_size // 2
        self.max_doc_length = config.get("max_doc_length", None)
        self.template = config.get("template", "pl")
        templates = {"pl": self._init_prompts_pl, "en": self._init_prompts_en, "apeer": self._init_prompts_apeer}
        init_prompts_func = templates.get(self.template)
        init_prompts_func()

    def _init_prompts_pl(self):
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

    def _init_prompts_en(self):
        self.system_prompt = (
            "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        )
        self.first_prompt = (
            "I will provide you with {num_docs} passages, each indicated by number identifier [].\n"
            "Rank the passages based on their relevance to query: {query}."
        )
        self.assistant_response = (
            "Okay, please provide the passages."
        )
        self.assistant_ack = (
            "Received passage [{num}]."
        )
        self.last_prompt = (
            "Search Query: {query}\n\n---\n\n"
            "Rank the passages above based on their relevance to the search query. "
            "The passages should be listed in descending order using identifiers. "
            "The most relevant passages should be listed first. "
            "The output format should be [] > [] > [], e.g., [1] > [2] > [3]. "
            "Only response the ranking results, do not say any word or explain."
        )

    def _init_prompts_apeer(self):
        self.system_prompt = (
            "As RankGPT, your task is to evaluate and rank unique passages based on their relevance and accuracy to a "
            "given query. Prioritize passages that directly address the query and provide detailed, correct answers. "
            "Ignore factors such as length, complexity, or writing style unless they seriously hinder readability."
        )
        self.first_prompt = (
            "In response to the query: [querystart] {query} [queryend], rank the passages. "
            "Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a "
            "comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the "
            "passages when determining their relevance.\n\n"
        )
        self.last_prompt = (
            "Given the query: [querystart] {query} [queryend], produce a succinct and clear ranking of all passages, "
            "from most to least relevant, using their identifiers. The format should be "
            "[rankstart] [most relevant passage ID] > [next most relevant passage ID] > ... > "
            "[least relevant passage ID] [rankend]. "
            "Refrain from including any additional commentary or explanations in your ranking."
        )

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        docs_copy = docs.copy()
        slice_start = len(docs) - self.sliding_window_size
        while slice_start >= 0:
            slice_end = slice_start + self.sliding_window_size
            docs_slice = docs_copy[slice_start:slice_end]
            sorted_slice = self.rerank_slice(query, docs_slice)
            for idx, slice_idx in enumerate(range(slice_start, slice_end)):
                doc = sorted_slice[idx]
                docs_copy[slice_idx] = doc
            slice_start -= self.sliding_window_overlap
        docs_scores = {doc: float(1.0 / (idx + 1.0)) for idx, doc in enumerate(docs_copy)}
        scores = [docs_scores[doc] for doc in docs]
        return scores

    def rerank_slice(self, query: str, docs: List[str]):
        messages = self.get_prompt(query, docs)
        request = LLMCall(messages)
        self.call(request)
        return self.create_ranks(request.response, docs)

    def rerank_many_slices(self, queries: List[str], docs: List[List[str]]):
        requests = [LLMCall(self.get_prompt(query, query_docs)) for query, query_docs in zip(queries, docs)]
        self.call_many_async(requests)
        return [self.create_ranks(req.response, req_docs) for req, req_docs in zip(requests, docs)]

    def create_ranks(self, response: str, docs: List[str]):
        response = "".join([(c if c.isdigit() else " ") for c in response]).strip()
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
        sorted_docs = [docs[i] for i in result]
        return sorted_docs

    def get_prompt(self, query: str, docs: List[str]):
        query = self._replace_number(query)
        docs = [self._replace_number(doc) for doc in docs]
        if self.template == "apeer":
            return self._get_prompt_apeer(query, docs)
        else:
            return self._get_prompt_multiturn(query, docs)

    def _get_prompt_multiturn(self, query: str, docs: List[str]):
        res = []
        res.append({"role": "system", "content": self.system_prompt})
        res.append({"role": "user", "content": self.first_prompt.format(num_docs=len(docs), query=query)})
        res.append({"role": "assistant", "content": self.assistant_response})
        for idx, doc in enumerate(docs):
            if self.max_doc_length is not None and len(doc) > self.max_doc_length:
                doc = doc[:self.max_doc_length - 5] + "(...)"
            res.append({"role": "user", "content": f"[{idx + 1}] {doc}"})
            res.append({"role": "assistant", "content": self.assistant_ack.format(num=idx + 1)})
        res.append({"role": "user", "content": self.last_prompt.format(query=query)})
        return res

    def _get_prompt_apeer(self, query: str, docs: List[str]):
        res = []
        res.append({"role": "system", "content": self.system_prompt})
        content = StringIO()
        content.write(self.first_prompt.format(query=query))
        for idx, doc in enumerate(docs):
            if self.max_doc_length is not None and len(doc) > self.max_doc_length:
                doc = doc[:self.max_doc_length - 5] + "(...)"
            content.write(f"[{idx + 1}] {doc}\n\n")
        content.write(self.last_prompt.format(query=query))
        res.append({"role": "user", "content": content.getvalue()})
        return res

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)


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


@dataclass
class Player:
    doc: str
    rating: Any
    matches: int = 0

    def skill(self):
        return self.rating.mu

    def conservative_skill(self, k: int = 3):
        return self.rating.mu - k * self.rating.sigma


class TrueskillReranker(RankGPTReranker):
    """
    Based on Microsoft Trueskill https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.max_players = config.get("max_players", 20)
        self.rounds = config.get("rounds", 10)

    def rerank(self, query: str, docs: List[str], proba: bool = False):
        import trueskill
        env = trueskill.TrueSkill(draw_probability=0.0)
        players: Dict[str, Player] = {}
        for doc in docs:
            rating = env.create_rating()
            players[doc] = Player(doc, rating)
        groups = self._initial_groups(docs)
        self.matches(query, env, players, groups)
        for i in range(self.rounds):
            players_list, docs_list = self.get_current_rating(players)
            #groups = [self.get_next_match(players_list, docs_list)]
            groups = self.get_next_round(docs_list)
            self.matches(query, env, players, groups)
        return [players[doc].rating.mu for doc in docs]

    def get_next_round(self, docs_list: List[str]):
        return [docs_list[i:i + self.max_players] for i in range(0, len(docs_list), self.max_players)]

    def get_next_match(self, players_list: List[Player], docs_list: List[str]):
        max_players = min(self.max_players, len(docs_list))
        top_players_num = math.ceil(0.25 * max_players)
        top_players = players_list[:max_players]
        top_players.sort(key=lambda p: p.rating.sigma, reverse=True)
        group = set()
        for idx in range(top_players_num):
            group.add(top_players[idx].doc)
        players_list.sort(key=lambda p: p.rating.sigma, reverse=True)
        for player in players_list:
            if player.doc in group:
                continue
            group.add(player.doc)
            if len(group) >= max_players:
                break
        return list(group)

    def get_current_rating(self, players: Dict[str, Player]):
        players_list = [player for player in players.values()]
        players_list.sort(key=lambda player: player.rating.mu, reverse=True)
        docs_list = [player.doc for player in players_list]
        return players_list, docs_list

    def _initial_groups(self, docs: List[str]):
        k = -(-len(docs) // self.max_players)
        matches = [[] for _ in range(k)]
        for i, elem in enumerate(docs):
            matches[i % k].append(elem)
        return matches

    def matches(self, query: str, env, players: Dict[str, Player], batch: List[List[str]]):
        calls = [LLMCall(self.get_prompt(query, docs)) for docs in batch]
        if len(calls) == 1:
            self.call(calls[0])
        else:
            self.call_many_async(calls)
        for idx, call in enumerate(calls):
            docs = batch[idx]
            sorted_docs = self.create_ranks(call.response, docs)
            sorted_players: List[Player] = [players[doc] for doc in sorted_docs]
            for player in sorted_players:
                player.matches += 1
            rated_players = env.rate([(player.rating,) for player in sorted_players])
            for k, new_rating in enumerate(rated_players):
                player = sorted_players[k]
                player.rating = new_rating[0]
