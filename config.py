import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the document processing system."""
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY_HERE")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "YOUR_PINECONE_ENVIRONMENT_HERE")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policy-index")

    DOCUMENTS_PATH: str = "./data/policies"
    PROCESSED_DOCS_PATH: str = "./data/processed"
    
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 1200  # Increased from 500 to preserve complete policy sections
    CHUNK_OVERLAP: int = 300  # Increased from 100 to 300 for better context preservation
    TOP_K_RESULTS: int = 8    # Increased from 4 to get more comprehensive context
    SIMILARITY_THRESHOLD: float = 0.5
    
    @classmethod
    def validate(cls) -> None:
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
            raise ValueError("PINECONE_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_ENVIRONMENT or cls.PINECONE_ENVIRONMENT == "YOUR_PINECONE_ENVIRONMENT_HERE":
            raise ValueError("PINECONE_ENVIRONMENT is not set. Please set the environment variable or update the config.py file.")

config = Config()