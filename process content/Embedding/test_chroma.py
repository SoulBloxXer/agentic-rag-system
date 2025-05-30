import chromadb

# Use the same persist directory as your main scripts
persist_dir = "process content/Embedding/embeddings"

client = chromadb.PersistentClient(path=persist_dir)

# Create or get the collection
collection = client.get_or_create_collection("test_collection")

# Add a dummy document
collection.add(
    documents=["hello world"],
    metadatas=[{"source": "test"}],
    ids=["test1"]
)

# Read back the documents
results = collection.get()
print(f"Document count in 'test_collection': {len(results['ids'])}")
print("Document IDs:", results['ids'])
print("Metadatas:", results['metadatas'])
print("Documents:", results['documents']) 