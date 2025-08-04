import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()


class Config:
    """Configuration class for the document processing system."""
    
    # OpenAI Configuration
    # IMPORTANT: Replace "YOUR_OPENAI_API_KEY_HERE" with your actual key.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    
    # Use gpt-3.5-turbo for a good balance of performance and cost. 
    # Change to gpt-4 or gpt-4o for better but more expensive results.
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # text-embedding-3-small is the recommended model for embeddings.
    # It's more cost-effective and performs better than text-embedding-ada-002.
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # File Paths
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    DOCUMENTS_PATH: str = "./data/policies"
    PROCESSED_DOCS_PATH: str = "./data/processed"
    
    # Processing Settings
    LOG_LEVEL: str = "INFO"
    
    # Chunking Settings - Optimized for RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # Retrieval Settings
    TOP_K_RESULTS: int = 4
    SIMILARITY_THRESHOLD: float = 0.5
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable or update the config.py file.")

# Global config instance
config = Config()