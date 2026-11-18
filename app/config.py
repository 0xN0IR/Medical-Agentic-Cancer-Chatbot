import os
import certifi
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get("SSL_CERT_FILE")
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()
elif not ssl_cert:
    os.environ["SSL_CERT_FILE"] = certifi.where()
# -------------------------------------------------------

load_dotenv()


class Settings(BaseSettings):
    gemini_api_key: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""

    pinecone_index_name: str = "cancer-agent-demo"
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    api_token: str = ""
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()