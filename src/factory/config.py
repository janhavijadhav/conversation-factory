import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider (works with any OpenAI-compatible API: Gemini, Groq, OpenAI)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# Model routing: each role can use a different model.
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")       # assistant = the training target
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", MODEL_NAME)        # scenario planner + simulated user
JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL_NAME)                # ideally a different model family

# Rate limiting: calls are spaced evenly to stay under the provider's limit.
# Applies per model client. 0 = no pacing.
REQUESTS_PER_MINUTE = 10

# Chain sampling
MIN_CHAIN_LENGTH = 2
MAX_CHAIN_LENGTH = 5

# Conversation limits (safety stops so nothing loops forever)
MAX_USER_TURNS = 5
MAX_TOOL_ROUNDS = 8

# Temperatures: creative for people, careful for the assistant, strict for the judge
PLANNER_TEMPERATURE = 0.9
USER_TEMPERATURE = 0.8
ASSISTANT_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.0
REWRITE_TEMPERATURE = 0.3

# Quality
JUDGE_PASS_THRESHOLD = 3.5   # mean of the 4 scores, out of 5
MIN_DIMENSION_SCORE = 3      # no single score may be below this
MAX_REPAIR_ATTEMPTS = 2

# Batching
BATCH_SIZE = 10
CONCURRENCY = 1              # one call at a time on the free tier
