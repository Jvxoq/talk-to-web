from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
QDRANT_HOST: str = os.environ["QDRANT_HOST"]
QDRANT_PORT: int = int(os.environ["QDRANT_PORT"])
DEEPGRAM_API_KEY: str = os.environ["DEEPGRAM_API_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]
