import os
import random
import grpc
import vector_store_pb2, vector_store_pb2_grpc
from dotenv import load_dotenv

SERVER_HOST = os.getenv("SERVER_HOST", "localhost:50051")

def seed_data(stub):
    items = [
        # Animals
        (1, "I love dogs."),
        (2, "Dogs are very loyal animals."),
        (3, "My puppy enjoys running in the park."),
        (4, "Cats are very independent."),
        (5, "I adopted a kitten yesterday."),
        (6, "The dog likes to jump."),
        (7, "The cat likes to jump."),
        (8, "A wolf is similar to a wild dog."),
        (9, "Lions are big cats."),
        (10, "Birds can fly high in the sky."),
        # Emotions / sentiment
        (20, "I am feeling very happy today."),
        (21, "This is the best day ever!"),
        (22, "I feel sad and tired."),
        (23, "Today has been a terrible day."),
        (24, "I am extremely excited about the future."),
        (25, "I feel depressed and lonely."),
        # Tech
        (30, "Python is a great programming language."),
        (31, "I enjoy writing backend services."),
        (32, "Distributed systems are fascinating."),
        (33, "Vector databases are useful for AI."),
        (34, "Machine learning powers modern applications."),
        (35, "I like building web apps with TypeScript."),
        # Actions
        (40, "He runs every morning."),
        (41, "She enjoys jogging at sunrise."),
        (42, "They sprinted across the field."),
        (43, "He walks slowly in the evening."),
        (44, "She strolled through the park."),
        (45, "They marched forward together."),
        # Random noise (realistic data)
        (60, "The weather is nice today."),
        (61, "I had pizza for lunch."),
        (62, "The movie was surprisingly good."),
        (63, "Coffee keeps me awake."),
        (64, "Music helps me focus."),
    ]

    for item_id, text in items:
        request = vector_store_pb2.UpsertRequest(id=item_id, text=text)
        response = stub.Upsert(request)
        print(f"Status: {response}")


def run_queries(stub):
    queries = [
        # Animal semantics
        "A dog jumping over a fence",
        "A playful puppy running outside",
        "Big wild cats in nature",
        "A kitten jumping in the house",
        # Sentiment
        "I feel extremely happy",
        "This is the worst day of my life",
        "I am excited and joyful",
        "I feel very depressed",
        # Tech semantics
        "AI embeddings and vector databases",
        "Backend programming and APIs",
        "Machine learning systems",
        "Building apps with Python",
        # Action similarity
        "Running very fast in the morning",
        "Walking slowly through a park",
        "People sprinting together",
        # Noise / mixed intent
        "Good weather and coffee",
        "Watching a great movie",
        "Relaxing with music",
    ]

    for query_text in queries:
        request = vector_store_pb2.SearchRequest(query_text=query_text, top_k=3)
        response = stub.Search(request)

        print("\n" + "=" * 60)
        print(f"Query: {query_text}")
        print(response)


def main():
    with grpc.insecure_channel(SERVER_HOST) as channel:
        stub = vector_store_pb2_grpc.VectorStoreStub(channel)

        seed_data(stub)

        print("Running semantic search tests...")
        run_queries(stub)


if __name__ == "__main__":
    main()
