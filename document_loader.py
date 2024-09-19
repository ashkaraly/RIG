# document_loader.py

# Import necessary libraries
from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
import pandas as pd

# Define Default Embedding Function
class DefChromaEF:
    def __init__(self, ef):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

# Create a Chroma collection if not exists
def setup_chroma_collection(client, collection_name="RIG"):
    if len(client.list_collections()) > 0 and collection_name in [client.list_collections()[0].name]:
        client.delete_collection(name=collection_name)
    else:
        print(f"Creating collection: '{collection_name}'")
        return client.create_collection(name=collection_name, embedding_function=embedding_functions.DefaultEmbeddingFunction(), metadata={"hnsw:space": "cosine"})

# Load data from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.info()
    return data

# Add documents and metadata to the Chroma collection
def add_to_collection(collection, data):
    collection.add(
        documents=data['Data_splits'].tolist(),
        ids=[str(index) for index in data.index.tolist()]
    )

# Main function to load documents
def main():
    # Initialize Chroma client (replace with actual client instance)
    chroma_client = None  # Define your Chroma client here
    collection_name = "RIG"
    
    # Setup Chroma collection
    talks_collection = setup_chroma_collection(chroma_client, collection_name)
    
    # Load data
    data = load_data("data.csv")
    
    # Add data to Chroma collection
    add_to_collection(talks_collection, data)

    print(f"Documents added to collection '{collection_name}'")

# Run the document loader
if __name__ == "__main__":
    main()
