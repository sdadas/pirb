from typing import Dict


class IndexBackend:

    @staticmethod
    def from_config(config: Dict):
        backend_type = config["backend_type"]
        index_dir = config["index_dir"]
        if backend_type == "faiss":
            from backend.in_memory import FaissBackend
            return FaissBackend(index_dir)
        elif backend_type == "raw":
            from backend.in_memory import RawVectorsBackend
            return RawVectorsBackend(index_dir)
        elif backend_type == "qdrant":
            from backend.qdrant import QdrantBackend
            host = config.get("host", "localhost")
            port = config.get("port", 6333)
            return QdrantBackend(index_dir, host=host, port=port)
        raise AssertionError("Unknown backend type: {}".format(backend_type))
