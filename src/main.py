from classes.vector_search_service import VectorSearchService

def main():
    service = VectorSearchService()

    service.add_item(1, "Hello World!")
    service.add_item(2, "Goodbye World!")
    service.add_item(3, "I love dogs!")
    service.add_item(4, "I hate dogs!")
    service.add_item(5, "The dog likes to jump.")
    service.add_item(69, "The cat likes to jump.")

    results = service.search("The dogs jumps over a big wall")

    print(results)

if __name__ == "__main__":
    main()
