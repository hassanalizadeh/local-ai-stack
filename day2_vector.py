import os
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load the environment variables from the .env file
load_dotenv()

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
QDRANT_PORT = int(os.getenv("QDRANT_REST_PORT", 6333))
QDRANT_HOST = "localhost" # Running locally, pointing to the mapped port
COLLECTION_NAME = "technical_docs"

# 1. Helper to get embeddings using the modern /api/embed endpoint
def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    payload = {
        "model": model,
        "input": text  # Changed from 'prompt' to 'input'
    }
    
    embed_url = f"{OLLAMA_API_BASE}/api/embed"
    response = httpx.post(embed_url, json=payload, timeout=30.0)
    response.raise_for_status()
    
    # Extract the first vector from the returned embeddings array
    return response.json()["embeddings"][0]

def main():
    print(f"🔌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Sample documents with metadata
    docs = [
        {"id": 1, "text": "Implement JWT validation using public keys.", "meta": {"topic": "auth", "lang": "python"}},
        {"id": 2, "text": "Optimize SQL queries by adding composite indices on foreign keys.", "meta": {"topic": "database", "lang": "sql"}},
        {"id": 3, "text": "Configure Redis eviction policies to volatile-lru for session caching.", "meta": {"topic": "cache", "lang": "redis"}}
    ]
    
    print("🧠 Extracting baseline embedding to verify vector dimensions...")
    # This will now use nomic-embed-text and succeed without the 500 error
    sample_vector = get_embedding(docs[0]["text"])
    vector_dimension = len(sample_vector)
    print(f"📐 Vector Space Dimension: {vector_dimension} (Nomic standard is usually 768)")
    
    # 2. Setup Qdrant Collection
    print(f"📁 Preparing Qdrant collection: '{COLLECTION_NAME}'...")
    
    # Safely check and delete to mimic the old "recreate" behavior
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        
    # Create the fresh collection
    client.create_collection(
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
    
    # NEW METHOD: query_points instead of search
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1
    ).points  # Make sure to append .points to get the list of hits
    
    for hit in search_results:
        print(f"🎯 Top Match (Score: {hit.score:.4f}):")
        print(f"   Text: {hit.payload['text']}")
        print(f"   Metadata: Topic={hit.payload['topic']}, Lang={hit.payload['lang']}")

if __name__ == "__main__":
    main()