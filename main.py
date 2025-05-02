#!/usr/bin/env python3
"""
RAGStackXL - Main entry point
"""
import os
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

from app.utils.logging import log
from app.config.settings import settings
from app.document_processing.processor import document_processor
from app.core.interfaces import RagDocument


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
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--query", "-q", required=True, help="Query to run")
    
    # Run server command
    server_parser = subparsers.add_parser("server", help="Run the server")
    server_parser.add_argument("--host", help=f"Host to bind (default: {settings.API.HOST})")
    server_parser.add_argument("--port", type=int, help=f"Port to bind (default: {settings.API.PORT})")
    
    return parser.parse_args()


def ingest_documents(
    source: str,
    recursive: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    splitter: Optional[str] = None,
) -> List[RagDocument]:
    """
    Ingest documents into the system.
    
    Args:
        source: Source file or directory to ingest.
        recursive: Whether to recursively ingest directories.
        chunk_size: Chunk size to use.
        chunk_overlap: Chunk overlap to use.
        splitter: Text splitter to use.
        
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
        
        log.info(f"Successfully ingested {len(documents)} document chunks")
        
        # TODO: Add vector storage integration
        log.warning("Vector storage integration is not yet implemented - documents are not being saved")
        
        return documents
    
    except Exception as e:
        log.error(f"Error ingesting documents: {e}")
        return []


def query_system(query: str):
    """Query the system."""
    log.info(f"Querying system with: {query}")
    # TODO: Implement querying
    log.info("Querying is not yet implemented")


def run_server(host: str = None, port: int = None):
    """Run the server."""
    host = host or settings.API.HOST
    port = port or settings.API.PORT
    
    log.info(f"Starting server on {host}:{port}")
    # TODO: Implement server
    log.info("Server is not yet implemented")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create required directories
    os.makedirs(settings.DOCUMENT.DATA_DIR, exist_ok=True)
    os.makedirs(settings.VECTORDB.PERSIST_DIRECTORY, exist_ok=True)
    
    log.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    log.info(f"Environment: {settings.ENVIRONMENT}")
    
    if args.command == "ingest":
        ingest_documents(
            source=args.source,
            recursive=args.recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            splitter=args.splitter,
        )
    elif args.command == "query":
        query_system(args.query)
    elif args.command == "server":
        run_server(args.host, args.port)
    else:
        log.error(f"Unknown command: {args.command}")
        sys.exit(1)
    
    log.info("Done")


if __name__ == "__main__":
    main() 