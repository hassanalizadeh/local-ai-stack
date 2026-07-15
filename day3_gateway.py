import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load the environment variables from the .env file
load_dotenv()

# Point to your LiteLLM Gateway container endpoint
client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key=os.getenv("LITELLM_MASTER_KEY") # LiteLLM proxy satisfies standard headers
)

# Define the structured schema target using Pydantic
class UserProfile(BaseModel):
    name: str = Field(description="The full name of the developer.")
    user_id: int = Field(description="A randomized unique 4-digit ID.")
    role: str = Field(description="The primary engineering focus role.")

def test_structured_gateway_call():
    print("📡 Route payload through LiteLLM Gateway...")
    
    try:
        # We call the generic alias 'local-llama' defined in litellm_config.yaml
        completion = client.beta.chat.completions.parse(
            model="local-llama",
            messages=[
                {"role": "system", "content": "You are a helpful system administrator."},
                {"role": "user", "content": "Generate a profile for a senior system engineer named Alex."}
            ],
            response_format=UserProfile,
            temperature=0.0 # Keep temperature low for rigid schema adhesion
        )
        
        parsed_profile = completion.choices[0].message.parsed
        print("\n✅ Gateway Response Successfully Parsed:")
        print(f"👤 Name: {parsed_profile.name}")
        print(f"🆔 ID:   {parsed_profile.user_id}")
        print(f"🛠️ Role: {parsed_profile.role}")
        
    except Exception as e:
        print(f"❌ Failed to proxy or parse model generation: {e}")

if __name__ == "__main__":
    test_structured_gateway_call()