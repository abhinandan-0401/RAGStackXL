#!/usr/bin/env python3
"""
RAGStackXL - Main entry point
"""
import os
import argparse
import sys
from pathlib import Path

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

from app.utils.logging import log
from app.config.settings import settings


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAGStackXL - Advanced RAG System with Agentic Capabilities")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest documents command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the system")
    ingest_parser.add_argument("--source", "-s", required=True, help="Source file or directory to ingest")
    ingest_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively ingest directories")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--query", "-q", required=True, help="Query to run")
    
    # Run server command
    server_parser = subparsers.add_parser("server", help="Run the server")
    server_parser.add_argument("--host", help=f"Host to bind (default: {settings.API.HOST})")
    server_parser.add_argument("--port", type=int, help=f"Port to bind (default: {settings.API.PORT})")
    
    return parser.parse_args()


def ingest_documents(source: str, recursive: bool = False):
    """Ingest documents into the system."""
    log.info(f"Ingesting documents from {source} (recursive: {recursive})")
    # TODO: Implement document ingestion
    log.info("Document ingestion is not yet implemented")


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
        ingest_documents(args.source, args.recursive)
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