import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Point to LiteLLM Gateway
client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key=os.getenv("LITELLM_MASTER_KEY")
)

def test_telemetry():
    print("📡 Sending request through Gateway to trigger Langfuse trace...")
    
    response = client.chat.completions.create(
        model="local-llama",
        messages=[
            {"role": "system", "content": "You are a database architect."},
            {"role": "user", "content": "Explain the difference between OLTP and OLAP in two sentences."}
        ],
        # Langfuse uses the 'user' field to track activity per individual
        user="dev-alex-local" 
    )
    
    print("\n✅ Response Received:")
    print(response.choices[0].message.content)
    print("\n🔍 Check http://localhost:3000 -> Traces to view the execution payload!")

if __name__ == "__main__":
    test_telemetry()