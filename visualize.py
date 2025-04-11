import chromadb
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pandas as pd

def load_chromadb_data(storage_path, collection_name):
    client = chromadb.PersistentClient(path=storage_path)
    collection = client.get_collection(collection_name)
    results = collection.get(include=["embeddings", "metadatas"])
    return results["embeddings"], results["metadatas"], results["ids"]

def reduce_dimensionality(embeddings, method="tsne", n_components=2):
    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Use 'tsne' or 'umap'.")
    return reducer.fit_transform(embeddings)

def visualize_embeddings(embeddings_2d, metadata, ids):
    names = [meta.get("name", "unknown") for meta in metadata]
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "name": names,
        "id": ids,
        "metadata": [str(meta) for meta in metadata]
    })
    fig = px.scatter(
        df, x="x", y="y", color="name",
        hover_data=["id", "metadata"],
        title="Face Recognition Embeddings in 2D"
    )
    fig.show()

def list_collections(storage_path):
    client = chromadb.PersistentClient(path=storage_path)
    print("Available collections:")
    for collection_name in client.list_collections():
        print(f"- {collection_name}")

def main():
    storage_path = "chroma_storage"
    list_collections(storage_path)
    collection_name = "faces"
    embeddings, metadata, ids = load_chromadb_data(storage_path, collection_name)
    reduced_embeddings = reduce_dimensionality(embeddings, method="tsne", n_components=2)
    visualize_embeddings(reduced_embeddings, metadata, ids)

if __name__ == "__main__":
    main()