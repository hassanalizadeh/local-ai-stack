# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the essential packages for the first 3 days
pip install httpx qdrant-client pydantic openai python-dotenv