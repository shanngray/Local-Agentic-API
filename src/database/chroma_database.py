import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

async def get_chroma_client():
    """Get a ChromaDB client."""
    return chromadb.PersistentClient(
        path="../../agent_memory_store",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
