import torch
import numpy as np
from .embedding_model import EmbeddingModel
from .vector_store import VectorStore


class VectorSearchService:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def add_item(self, item_id: int, text: str):
        embedding = self.embedding_model.encode(text)
        self.vector_store.upsert(item_id, text, embedding)

    def search(self, text, top_k=3):
        embedding = self.embedding_model.encode(text)

        ids = list(self.vector_store.store.keys())
        embeddings = np.array(list(self.vector_store.store.values()))

        similarities = self.embedding_model.similarity(embedding, embeddings)
        top_results = torch.argsort(similarities, descending=True)[0][:top_k]

        return [(self.vector_store.metadata[ids[i]]) for i in top_results]
