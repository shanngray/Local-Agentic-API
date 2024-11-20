from typing import Optional
from pydantic import BaseModel

# ... existing configurations ...

class APIConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    log_level: str = "INFO"
    # Add any additional configurations related to conversations if needed

    class Config:
        env_prefix = "API_" 

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    
    # Add any other configuration settings here
    DEBUG: bool = False
    MAX_TURNS: int = 5

settings = Settings()
