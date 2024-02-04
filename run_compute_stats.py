import logging

from data import RawJsonlTask, MAUPQATask, PolEvalTask, BEIRTask, MFAQTask, GPTExamsTask

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    benchmark = [
        # Web crawled datasets
        RawJsonlTask("gemini"),
        RawJsonlTask("odi"),
        RawJsonlTask("onet"),
        RawJsonlTask("zapytajfizyka"),
        RawJsonlTask("techpedia"),
        RawJsonlTask("pwn"),
        RawJsonlTask("eprawnik"),
        RawJsonlTask("specprawnik"),
        RawJsonlTask("abczdrowie"),
        # MAUPQA datasets
        MAUPQATask("1z10"),
        MAUPQATask("czy-wiesz-v2"),
        MAUPQATask("gpt3-cc"),
        MAUPQATask("gpt3.5-cc"),
        MAUPQATask("gpt3.5-wiki"),
        MAUPQATask("mkqa"),
        MAUPQATask("mqa"),
        MAUPQATask("multilingual-NLI"),
        MAUPQATask("poleval2021-pairs"),
        MAUPQATask("poquad"),
        MAUPQATask("templates"),
        MAUPQATask("wiki-def"),
        # PolEval-2022
        PolEvalTask("dev-0", "wiki-trivia"),
        PolEvalTask("test-A", "wiki-trivia"),
        PolEvalTask("test-A", "legal-questions"),
        PolEvalTask("test-A", "allegro-faq"),
        PolEvalTask("test-B", "wiki-trivia"),
        PolEvalTask("test-B", "legal-questions"),
        PolEvalTask("test-B", "allegro-faq"),
        # BEIR-PL datasets
        BEIRTask("arguana-pl", skip_self=True),
        BEIRTask("dbpedia-pl"),
        BEIRTask("fiqa-pl"),
        BEIRTask("hotpotqa-pl"),
        BEIRTask("msmarco-pl", splits=("validation",)),
        BEIRTask("nfcorpus-pl"),
        BEIRTask("nq-pl"),
        BEIRTask("quora-pl", skip_self=True),
        BEIRTask("scidocs-pl"),
        BEIRTask("scifact-pl"),
        BEIRTask("trec-covid-pl"),
        # Other datasets
        MFAQTask(),
        GPTExamsTask()
    ]

    for task in benchmark:
        task.compute_stats("data")