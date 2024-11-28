from custom_faiss.faiss_manager import FaissManager

# Global singleton instance
_faiss_manager = None

def get_faiss_manager(embedder_name, distance_metric):
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FaissManager(embedder_name, distance_metric)
    return _faiss_manager
