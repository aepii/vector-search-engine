import os
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from classes.vector_service import VectorService
from dotenv import load_dotenv
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

load_dotenv()

SERVER_PORT = os.getenv("SERVER_PORT", "50051")


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """gRPC servicer implementation for the VectorStore service."""

    def __init__(self):
        self.service = VectorService()

    def Upsert(self, request, context):
        """
        Inserts or updates an item in the vector store.
        """
        item = request.item
        self.service.add_item(item.id, item.text)
        return vector_store_pb2.UpsertResponse(status=f"ID {request.id} indexed.")

    def UpsertBatch(self, request, context):
        """
        Inserts or updates a batch of items in the vector store.
        """
        items = [(item.id, item.text) for item in request.items]
        self.service.add_items_batch(items)
        statuses = [f"ID {item.id} indexed." for item in request.items]
        return vector_store_pb2.UpsertBatchResponse(statuses=statuses)

    def Search(self, request, context):
        """
        Performs a similarity search over the indexed vectors.
        """
        results = self.service.search(request.query_text, top_k=request.top_k)
        return vector_store_pb2.SearchResponse(
            results=[
                vector_store_pb2.SearchResult(text=text, score=score)
                for text, score in results
            ]
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(), server
    )
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    print(f"Vector Store Server started on port {SERVER_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
