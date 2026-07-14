import httpx
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the variables using os.getenv(key, default_value)
ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

OLLAMA_EMBED_URL = ollama_base + "/api/embeddings"
QDRANT_PORT = os.getenv("QDRANT_REST_PORT")
QDRANT_HOST = "localhost"
COLLECTION_NAME = "technical_docs"

# 1. Helper to get embeddings from native Ollama
def get_embedding(text: str, model: str = "qwen2.5-coder:14b") -> list[float]:
    response = httpx.post(OLLAMA_EMBED_URL, json={"model": model, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def main():
    # Initialize Qdrant Client pointing to Docker container
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Sample documents with metadata
    docs = [
        {"id": 1, "text": "Implement JWT validation using public keys.", "meta": {"topic": "auth", "lang": "python"}},
        {"id": 2, "text": "Optimize SQL queries by adding composite indices on foreign keys.", "meta": {"topic": "database", "lang": "sql"}},
        {"id": 3, "text": "Configure Redis eviction policies to volatile-lru for session caching.", "meta": {"topic": "cache", "lang": "redis"}}
    ]
    
    print("🧠 Extracting baseline embedding to verify vector dimensions...")
    sample_vector = get_embedding(docs[0]["text"])
    vector_dimension = len(sample_vector)
    print(f"📐 Vector Space Dimension: {vector_dimension}")
    
    # 2. Setup Qdrant Collection
    print(f"📁 Re-creating Qdrant collection: '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
    )
    
    # 3. Populate Points with Payloads
    print("📥 Ingesting documents into vector store...")
    points = []
    for doc in docs:
        vector = get_embedding(doc["text"])
        points.append(
            PointStruct(
                id=doc["id"],
                vector=vector,
                payload={"text": doc["text"], **doc["meta"]}
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Ingestion complete.")
    
    # 4. Semantic Query Validation
    query_phrase = "How do I secure my endpoints and check user identity?"
    print(f"\n🔍 Performing semantic search for: '{query_phrase}'")
    query_vector = get_embedding(query_phrase)
    
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=1
    )
    
    for hit in search_results:
        print(f"🎯 Top Match (Score: {hit.score:.4f}):")
        print(f"   Text: {hit.payload['text']}")
        print(f"   Metadata: Topic={hit.payload['topic']}, Lang={hit.payload['lang']}")

if __name__ == "__main__":
    main()