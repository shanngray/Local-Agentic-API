'''
ChromaDB CLI Tool
python -m src.utils.chroma_cli --list
python -m src.utils.chroma_cli --show <collection_name>
'''
import asyncio
import argparse
from typing import Optional
from pathlib import Path

from src.database.chroma_database import get_chroma_client

async def list_collections(client) -> None:
    """Display all collections with their statistics."""
    collections = client.list_collections()
    
    if not collections:
        print("No collections found.")
        return

    # Gather stats for each collection
    collection_stats = []
    for col in collections:
        count = col.count()
        peek = col.peek()
        collection_stats.append({
            "Name": col.name,
            "Documents": count,
            "Sample Fields": list(peek["metadatas"][0].keys()) if count > 0 and peek["metadatas"] else []
        })

    # Determine column widths
    widths = {
        "Name": max(len("Collection Name"), max(len(str(stats["Name"])) for stats in collection_stats)),
        "Documents": max(len("Document Count"), max(len(str(stats["Documents"])) for stats in collection_stats)),
        "Sample Fields": max(len("Metadata Fields"), max(len(", ".join(stats["Sample Fields"])) for stats in collection_stats))
    }

    # Create format string
    fmt = "| {:<{}} | {:^{}} | {:<{}} |"
    separator = "-" * (sum(widths.values()) + 7)  # +7 for the borders and spaces

    # Print header
    print("\nChromaDB Collections:")
    print(separator)
    print(fmt.format("Collection Name", widths["Name"],
                    "Document Count", widths["Documents"],
                    "Metadata Fields", widths["Sample Fields"]))
    print(separator)

    # Print data
    for stats in collection_stats:
        print(fmt.format(
            stats["Name"], widths["Name"],
            stats["Documents"], widths["Documents"],
            ", ".join(stats["Sample Fields"]) if stats["Sample Fields"] else "N/A", widths["Sample Fields"]
        ))
    print(separator + "\n")

async def show_collection(client, collection_name: str) -> None:
    """Display all documents and metadata for a specific collection."""
    try:
        collection = client.get_collection(collection_name)
        results = collection.get()
        
        if not results["ids"]:
            print(f"\nNo documents found in collection: {collection_name}")
            return

        # Determine column widths
        id_width = max(len("ID"), max(len(str(id_)) for id_ in results["ids"]))
        content_width = max(len("Content"), max(len(str(doc)[:100]) for doc in results["documents"]))
        metadata_width = max(len("Metadata"), max(len(str(meta)) for meta in results["metadatas"]))

        # Create format string
        fmt = "| {:<{}} | {:<{}} | {:<{}} |"
        separator = "-" * (id_width + content_width + metadata_width + 7)

        # Print header
        print(f"\nDocuments in collection: {collection_name}")
        print(separator)
        print(fmt.format("ID", id_width, "Content", content_width, "Metadata", metadata_width))
        print(separator)

        # Print data
        for id_, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            content = (doc[:97] + "...") if len(doc) > 100 else doc
            print(fmt.format(
                str(id_), id_width,
                content, content_width,
                str(metadata), metadata_width
            ))
        print(separator + "\n")
            
    except Exception as e:
        print(f"Error accessing collection {collection_name}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="ChromaDB CLI Tool")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--show", type=str, help="Show documents in specified collection")
    
    args = parser.parse_args()
    
    # Get ChromaDB client
    client = await get_chroma_client()
    
    if args.list:
        await list_collections(client)
    elif args.show:
        await show_collection(client, args.show)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
