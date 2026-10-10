"""Print every model your API key can use."""
from openai import OpenAI
from src.factory.config import LLM_API_KEY, LLM_BASE_URL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
for m in sorted(client.models.list().data, key=lambda m: m.id):
    print(m.id)
