import os
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

# Configuration
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY")
COLLECTION_NAME = "technical_docs"

qdrant = QdrantClient(host="localhost", port=6333)
llm_client = OpenAI(base_url="http://localhost:4000/v1", api_key=LITELLM_KEY)

def get_embedding(text: str) -> list[float]:
    # embed_url = f"{OLLAMA_API_BASE}/api/embed"
    # response = httpx.post(embed_url, json={"model": "nomic-embed-text", "prompt": text})
    response = llm_client.embeddings.create(
        model="local-embed",
        input=[text] # Accepts an array of strings to process multiple items at once
    )
    return response.data[0].embedding

def retrieve_context(query: str) -> str:
    query_vector = get_embedding(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=1
    )
    if results:
        return results[0].payload["text"]
    return "No context found."

def full_pipeline(user_query: str):
    print(f"🔍 Searching Qdrant for: '{user_query}'...")
    context = retrieve_context(user_query)
    print(f"📄 Found Context: {context}")
    
    print("\n🧠 Sending augmented prompt to LiteLLM...")
    prompt = f"Use this context to answer the question: {context}\n\nQuestion: {user_query}"
    
    response = llm_client.chat.completions.create(
        model="local-llama",
        messages=[{"role": "user", "content": prompt}],
    )
    
    print(f"\n🤖 Final Answer: {response.choices[0].message.content}")

if __name__ == "__main__":
    # If you run this twice, the second time should be instant due to Redis caching!
    full_pipeline("How do I secure my endpoints and check user identity?")
