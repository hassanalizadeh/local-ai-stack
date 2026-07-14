import time
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

def benchmark_inference():
    payload = {
        "model": "llama3",
        "prompt": "Return the exact phrase: Engine is online.",
        "stream": False
    }
    
    print("⏳ Sending request to native Ollama...")
    start_time = time.perf_counter()
    
    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        response.raise_for_status()
        
        duration = time.perf_counter() - start_time
        result = response.json()
        
        print("\n✅ Success!")
        print(f"🤖 Model Response: {result.get('response').strip()}")
        print(f"⏱️ Total Latency: {duration:.4f} seconds")
        
    except httpx.HTTPError as e:
        print(f"❌ HTTP Request failed: {e}")

if __name__ == "__main__":
    benchmark_inference()