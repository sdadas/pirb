from typing import Dict, Union
from backend.base import DenseBackend, SparseBackend, LateInteractionBackend


class IndexBackend:

    @staticmethod
    def from_config(config: Dict) -> Union[DenseBackend, SparseBackend, LateInteractionBackend]:
        backend_type = config["backend_type"]
        index_dir = config["index_dir"]
        if backend_type == "faiss":
            from backend.dense import FaissBackend
            return FaissBackend(index_dir)
        elif backend_type == "raw":
            from backend.dense import RawVectorsBackend
            return RawVectorsBackend(index_dir)
        elif backend_type == "qdrant":
            from backend.dense_qdrant import QdrantBackend
            host = config.get("host", "localhost")
            port = config.get("port", 6333)
            return QdrantBackend(index_dir, host=host, port=port)
        elif backend_type == "sparse_csr":
            from backend.sparse_csr import SparseCSRBackend
            return SparseCSRBackend(index_dir)
        elif backend_type == "seismic":
            from backend.sparse_seismic import SparseSeismicBackend
            return SparseSeismicBackend(index_dir)
        elif backend_type == "plaid":
            from backend.late_interaction import PlaidBackend
            return PlaidBackend(index_dir)
        elif backend_type == "voyager":
            from backend.late_interaction import VoyagerBackend
            return VoyagerBackend(index_dir)
        raise AssertionError("Unknown backend type: {}".format(backend_type))
