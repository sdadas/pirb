## Polish Information Retrieval Benchmark

This repository contains the evaluation code for Polish Information Retrieval Benchmark (PIRB). 
The benchmark covers 41 Polish multidomain information retrieval tasks. 
Its purpose is to evaluate Polish and multilingual information retrieval methods on a wide range of problems with different characteristics, thus testing the generalization ability of the models and their zero-shot performance. 
It includes pre-existing datasets such as MaupQA, BEIR-PL and PolEval-2022. 
We have also added new, previously unpublished datasets. The "Web Datasets" group contains real questions and answers from Polish web services.

⚠️ This code should be run on Linux. We noticed problems with methods based on anserini returning incorrect results on Windows hosts. ⚠️

### 1. How to run the evaluation?

To evaluate a model or a list of models on PIRB, use `run_benchmark.py` script. 
The only required parameter for the script is `--models_config`, which should point to a json file containing a configuration of the models.
The repository supports many text retrieval methods, including sparse and dense retrievers, hybrid retrieval, as well as two-stage retrieval pipelines combining retriever and reranker models.
The configuration file should be a json array in which each element defines one method to be evaluated.
For example, below is the simplest configuration which defines a BM25 baseline:

```json
[{"name": "bm25"}]
```

Dense encoders based on Sentence-Transformers library can be defined in the following way:

```json
[
  {
    "name": "sdadas/mmlw-e5-base",
    "fp16": true,
    "q_prefix": "query: ",
    "p_prefix": "passage: "
  }
]
```

The `name` attribute should refer to a local path or path on the Huggingface Hub. 
Other attributes are optional, and allow to control the behavior of the model.
Methods combining multiple models require more complex configuration.
Below is the example of two-stage retrieval system with dense retriever and T5-based reranker:

```json
[
  {
    "name": "plt5-large-msmarco",
    "type": "hybrid",
    "k0": 100,
    "strategy": {
      "type": "reranker",
      "reranker_name": "clarin-knext/plt5-large-msmarco",
      "reranker_type": "seq2seq",
      "batch_size": 32,
      "max_seq_length": 512,
      "template": "Query: {query} Document: {passage} Relevant:",
      "yes_token": "prawda",
      "no_token": "fałsz",
      "bf16": true
    },
    "models": [
      {
        "name": "sdadas/mmlw-retrieval-roberta-large",
        "fp16": true,
        "q_prefix": "zapytanie: "
      }
    ]
  }
]
```

More examples of method definitions can be found in the `config` directory in this repository.

### 2. How to obtain the datasets?

Most of the data used in the evaluation is publicly available. The datasets will be automatically downloaded upon the first run of the `run_benchmark.py` script. The only exception are the corpora from the "Web Datasets" group. If you would like to access them, please send a request to sdadas at opi.org.pl, describing your intended use of the datasets. Please note that the datasets can only be used for research purposes and we request not to redistribute them after obtaining access.

### 3. How to add my model to the leaderboard?

If you have a model that has not been included in the ranking yet, open a new issue at https://huggingface.co/spaces/sdadas/pirb/discussions with a description of your model. 
We will try to evaluate it and add it to the leaderboard. 
In the description you can include a json configuration for the model in the PIRB format or a short code fragment illustrating the use of the model. 
In the official evaluation, we only consider models that:
<br/>
1\. Are publicly available
<br/>
2\. Have not been trained on the data sources included in PIRB. For datasets split into train, eval and test parts, the use of the training split is acceptable.