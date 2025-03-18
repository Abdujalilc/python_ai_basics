from custom_faiss.faiss_manager_singleton import get_faiss_manager

def update_faiss_manager(embedder_name, distance_metric):
    faiss_manager = get_faiss_manager(embedder_name,distance_metric)
    faiss_manager.update_parameters(embedder_name, distance_metric)
