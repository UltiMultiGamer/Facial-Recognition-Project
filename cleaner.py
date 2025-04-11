import chromadb

# Step 1: Initialize ChromaDB and Load Data
def load_chromadb_data(storage_path, collection_name):
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=storage_path)

    # Access the collection
    collection = client.get_collection(collection_name)

    # Query the collection to get all embeddings, metadata, and IDs
    results = collection.get(include=["embeddings", "metadatas"])

    # Extract embeddings, metadata, and IDs
    embeddings = results["embeddings"]
    metadata = results["metadatas"]
    ids = results["ids"]  # IDs are automatically included in the results

    return collection, embeddings, metadata, ids

# Step 2: Delete Specific Points/Vectors
def delete_vectors(collection, ids_to_delete):
    # Delete vectors by their IDs
    collection.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} vectors with IDs: {ids_to_delete}")

# Step 3: Main Function
def main():
    # Path to the ChromaDB storage folder
    storage_path = "chroma_storage"

    # ChromaDB collection name
    collection_name = "faces"

    # Load data from ChromaDB
    collection, embeddings, metadata, ids = load_chromadb_data(storage_path, collection_name)



    # Ask the user which IDs to delete
    ids_to_delete = input("Enter the IDs to delete (comma-separated): ").strip().split(",")
    ids_to_delete = [id.strip() for id in ids_to_delete]  # Clean up input

    # Delete the specified vectors
    delete_vectors(collection, ids_to_delete)

if __name__ == "__main__":
    main()