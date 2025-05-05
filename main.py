#!/usr/bin/env python3
"""
RAGStackXL - Main entry point
"""
import os
import argparse
import sys
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

from app.utils.logging import log
from app.config.settings import settings
from app.document_processing.processor import document_processor
from app.core.interfaces import RagDocument
from app.vectordb import create_vectordb
from app.vectordb.utils import add_vectordb_args, create_vectordb_from_args, format_search_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAGStackXL - Advanced RAG System with Agentic Capabilities")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest documents command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the system")
    ingest_parser.add_argument("--source", "-s", required=True, help="Source file or directory to ingest")
    ingest_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively ingest directories")
    ingest_parser.add_argument("--chunk-size", type=int, help=f"Chunk size (default: {settings.DOCUMENT.CHUNK_SIZE})")
    ingest_parser.add_argument("--chunk-overlap", type=int, help=f"Chunk overlap (default: {settings.DOCUMENT.CHUNK_OVERLAP})")
    ingest_parser.add_argument("--splitter", help="Text splitter to use (RecursiveSemanticSplitter, SemanticTextSplitter, CharacterTextSplitter)")
    # Add vector database arguments
    add_vectordb_args(ingest_parser)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--query", "-q", required=True, help="Query to run")
    query_parser.add_argument("--k", type=int, default=4, help="Number of results to return")
    query_parser.add_argument("--filter", help="Filter query in JSON format (e.g., '{\"source\":\"example.pdf\"}')")
    query_parser.add_argument("--embedding-file", help="File containing pre-computed query embedding")
    # Add vector database arguments
    add_vectordb_args(query_parser)
    
    # Run server command
    server_parser = subparsers.add_parser("server", help="Run the server")
    server_parser.add_argument("--host", help=f"Host to bind (default: {settings.API.HOST})")
    server_parser.add_argument("--port", type=int, help=f"Port to bind (default: {settings.API.PORT})")
    # Add vector database arguments
    add_vectordb_args(server_parser)
    
    # Add database management commands
    db_parser = subparsers.add_parser("db", help="Database management commands")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database command to run")
    
    # Clear database command
    clear_parser = db_subparsers.add_parser("clear", help="Clear the database")
    add_vectordb_args(clear_parser)
    
    # Stats command
    stats_parser = db_subparsers.add_parser("stats", help="Get database statistics")
    add_vectordb_args(stats_parser)
    
    return parser.parse_args()


async def ingest_documents(
    source: str,
    recursive: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    splitter: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> List[RagDocument]:
    """
    Ingest documents into the system.
    
    Args:
        source: Source file or directory to ingest.
        recursive: Whether to recursively ingest directories.
        chunk_size: Chunk size to use.
        chunk_overlap: Chunk overlap to use.
        splitter: Text splitter to use.
        args: Command line arguments for additional settings.
        
    Returns:
        List of processed documents.
    """
    log.info(f"Ingesting documents from {source} (recursive: {recursive})")
    
    try:
        # Process documents
        documents = document_processor.process_document(
            source=source,
            splitter_name=splitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=recursive,
        )
        
        log.info(f"Successfully processed {len(documents)} document chunks")
        
        if args and len(documents) > 0:
            # Create vector database
            vector_db = create_vectordb_from_args(args)
            
            # TODO: Generate embeddings (for now using dummy embeddings)
            # In the next phase, we'll implement proper embedding generation
            dummy_dim = vector_db.config.embedding_dimension
            dummy_embeddings = [[0.0] * dummy_dim for _ in range(len(documents))]
            
            # Store documents in vector database
            doc_ids = await vector_db.add_documents(documents, dummy_embeddings)
            log.info(f"Added {len(doc_ids)} documents to vector database")
            
            # Get stats
            stats = await vector_db.get_collection_stats()
            log.info(f"Vector database stats: {stats}")
        
        return documents
    
    except Exception as e:
        log.error(f"Error ingesting documents: {e}")
        return []


async def query_system(
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    embedding_file: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
):
    """
    Query the system.
    
    Args:
        query: Query string.
        k: Number of results to return.
        filter: Filter to apply to results.
        embedding_file: File containing pre-computed query embedding.
        args: Command line arguments for additional settings.
    """
    log.info(f"Querying system with: {query}")
    
    try:
        # Create vector database
        vector_db = create_vectordb_from_args(args)
        
        # Get database stats
        stats = await vector_db.get_collection_stats()
        log.info(f"Vector database stats: {stats}")
        
        if stats.get("count", 0) == 0:
            log.warning("Vector database is empty. Ingest documents first.")
            return
        
        # TODO: Generate proper embeddings (for now using dummy data)
        # In the next phase, we'll implement proper embedding generation
        dummy_dim = vector_db.config.embedding_dimension
        query_embedding = [0.1] * dummy_dim
        
        # Perform search
        results = await vector_db.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        # Display results
        log.info(f"Found {len(results)} results:")
        log.info(format_search_results(results))
        
    except Exception as e:
        log.error(f"Error querying system: {e}")


async def run_server(host: str = None, port: int = None, args: Optional[argparse.Namespace] = None):
    """Run the server."""
    host = host or settings.API.HOST
    port = port or settings.API.PORT
    
    log.info(f"Starting server on {host}:{port}")
    # TODO: Implement server
    log.info("Server is not yet implemented")


async def manage_database(db_command: str, args: argparse.Namespace):
    """
    Manage the vector database.
    
    Args:
        db_command: Database command to run.
        args: Command line arguments.
    """
    # Create vector database
    vector_db = create_vectordb_from_args(args)
    
    if db_command == "clear":
        log.info("Clearing vector database...")
        success = await vector_db.clear_collection()
        if success:
            log.info("Vector database cleared successfully")
        else:
            log.error("Failed to clear vector database")
    
    elif db_command == "stats":
        log.info("Getting vector database statistics...")
        stats = await vector_db.get_collection_stats()
        log.info(f"Vector database stats: {stats}")
    
    else:
        log.error(f"Unknown database command: {db_command}")


async def main_async():
    """Async main entry point."""
    args = parse_args()
    
    # Create required directories
    os.makedirs(settings.DOCUMENT.DATA_DIR, exist_ok=True)
    os.makedirs(settings.VECTORDB.PERSIST_DIRECTORY, exist_ok=True)
    
    log.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    log.info(f"Environment: {settings.ENVIRONMENT}")
    
    if args.command == "ingest":
        await ingest_documents(
            source=args.source,
            recursive=args.recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            splitter=args.splitter,
            args=args
        )
    elif args.command == "query":
        filter_dict = None
        if args.filter:
            import json
            try:
                filter_dict = json.loads(args.filter)
            except json.JSONDecodeError:
                log.error(f"Invalid filter JSON: {args.filter}")
                return
        
        await query_system(
            query=args.query,
            k=args.k,
            filter=filter_dict,
            embedding_file=args.embedding_file,
            args=args
        )
    elif args.command == "server":
        await run_server(args.host, args.port, args)
    elif args.command == "db":
        if not args.db_command:
            log.error("No database command specified")
            return
        await manage_database(args.db_command, args)
    else:
        log.error(f"Unknown command: {args.command}")
        sys.exit(1)
    
    log.info("Done")


def main():
    """Main entry point."""
    # Create event loop and run main_async
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main() 