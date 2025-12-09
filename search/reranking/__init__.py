from search.reranking.base import *
from search.reranking.impl_common import *
from search.reranking.impl_llm import *


class Reranker:

    @staticmethod
    def from_config(config: Dict) -> RerankerBase:
        reranker_type = config.get("reranker_type", "classifier")
        if reranker_type == "classifier":
            reranker = ClassifierReranker(**config)
        elif reranker_type == "vllm":
            reranker = VLLMClassifierReranker(**config)
        elif reranker_type == "seq2seq":
            reranker = Seq2SeqReranker(**config)
        elif reranker_type in ("flag_classifier", "flag_llm", "flag_layerwise_llm", "flag_lightweight_llm"):
            reranker = FlagReranker(**config)
        elif reranker_type == "jina":
            reranker = JinaReranker(**config)
        elif reranker_type == "mixedbread":
            reranker = MixedbreadReranker(**config)
        elif reranker_type == "qwen3":
            reranker = Qwen3Reranker(**config)
        elif reranker_type == "ctxl":
            reranker = CtxlReranker(**config)
        elif reranker_type == "pylate":
            reranker = PylateReranker(**config)
        elif reranker_type == "llm_rankgpt":
            reranker = RankGPTReranker(**config)
        elif reranker_type == "llm_pairwise":
            reranker = PairwiseLLMReranker(**config)
        elif reranker_type == "llm_trueskill":
            reranker = TrueskillReranker(**config)
        elif reranker_type == "cross_encoder":
            reranker = CrossEncoderReranker(**config)
        else:
            raise AssertionError(f"Unknown reranker type {reranker_type}")
        return reranker
