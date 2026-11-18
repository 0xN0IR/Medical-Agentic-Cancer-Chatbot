import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

cert_path = certifi.where()
if not os.environ.get("SSL_CERT_FILE") or not Path(os.environ["SSL_CERT_FILE"]).exists():
    os.environ["SSL_CERT_FILE"] = cert_path
if not os.environ.get("REQUESTS_CA_BUNDLE") or not Path(os.environ["REQUESTS_CA_BUNDLE"]).exists():
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path

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