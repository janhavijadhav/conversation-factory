"""Send calls at a steady pace and report exactly when and why the limit hits."""
import time

from openai import OpenAI
from src.factory.config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME, REQUESTS_PER_MINUTE

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
gap = 60 / REQUESTS_PER_MINUTE
print(f"Model: {MODEL_NAME} | sending 8 calls, one every {gap:.0f}s\n")

for i in range(1, 9):
    start = time.time()
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        print(f"Call {i}: OK")
    except Exception as e:
        print(f"Call {i}: FAILED\n")
        print(str(e))
        break
    time.sleep(max(0, gap - (time.time() - start)))
