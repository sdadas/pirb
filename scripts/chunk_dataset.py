import json
import logging
import os
import shutil
from dataclasses import field, dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import HfArgumentParser


@dataclass
class ChunkingArgs:
    task_name: str = field(
        metadata={"help": "Name of the input directory containing queries and passages"},
    )
    output_task_name: str = field(
        metadata={"help": "Name of the output directory for chunked task"},
    )
    data_dir: str = field(
        default="../data",
        metadata={"help": "Directory where task data is stored"},
    )
    chunk_size: int = field(
        default=1000,
        metadata={"help": "Number of characters in a chunk"},
    )
    chunk_overlap: int = field(
        default=0,
        metadata={"help": "Token overlap between chunks"},
    )


class DatasetChunker:

    def __init__(self, args: ChunkingArgs):
        self.args = args

    def chunk(self):
        input_dir = os.path.join(self.args.data_dir, self.args.task_name)
        output_dir = os.path.join(self.args.data_dir, self.args.output_task_name)
        os.makedirs(output_dir, exist_ok=False)
        self._copy_data(input_dir, output_dir)
        self._chunk_documents(output_dir)

    def _chunk_documents(self, output_dir: str):
        logging.info("Chunking documents")
        passages_dir = os.path.join(output_dir, "passages")
        docs_path = os.path.join(output_dir, "documents/documents.jsonl")
        passages_path = os.path.join(output_dir, "passages/passages.jsonl")
        os.makedirs(passages_dir, exist_ok=True)
        num_docs, num_chunks = 0, 0
        split = RecursiveCharacterTextSplitter(chunk_size=self.args.chunk_size, chunk_overlap=self.args.chunk_overlap)
        with open(docs_path, "r", encoding="utf-8") as infile, open(passages_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                doc = json.loads(line)
                text = doc["contents"]
                num_docs += 1
                for idx, chunk in enumerate(split.split_text(text)):
                    new_doc = dict(doc)
                    new_doc["contents"] = chunk
                    if "parentId" not in new_doc:
                        new_doc["parentId"] = new_doc["id"]
                    new_doc["id"] = f"{new_doc['id']}-{idx + 1}"
                    outfile.write(json.dumps(new_doc, ensure_ascii=False))
                    outfile.write("\n")
                    num_chunks += 1
        logging.info("Created %d chunks from %d documents", num_chunks, num_docs)

    def _copy_data(self, input_dir: str, output_dir: str):
        shutil.copytree(os.path.join(input_dir, "queries"), os.path.join(output_dir, "queries"))
        documents_dir = os.path.join(input_dir, "documents")
        passages_dir = os.path.join(input_dir, "passages")
        if os.path.exists(documents_dir):
            shutil.copytree(documents_dir, os.path.join(output_dir, "documents"))
        else:
            output_documents_dir = os.path.join(output_dir, "documents")
            os.makedirs(output_documents_dir, exist_ok=True)
            shutil.copy(
                os.path.join(passages_dir, "passages.jsonl"),
                os.path.join(output_documents_dir, "documents.jsonl")
            )


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser([ChunkingArgs])
    args = parser.parse_args_into_dataclasses()[0]
    chunker = DatasetChunker(args)
    chunker.chunk()
