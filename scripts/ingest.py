"""Script to ingest documents into the vector store."""

import os
import sys

# Add project root to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.core import logger, setup_logging
from src.ingestion import ingest_documents, load_documents, split_documents

setup_logging("INFO")


def main() -> None:
    """Run document ingestion and display chunk information."""
    logger.info("Starting document ingestion...")
    logger.info("=" * 60)
    
    # Load documents
    docs = load_documents()
    logger.info(f"Loaded {len(docs)} documents from URLs")
    
    # Display document metadata
    for i, doc in enumerate(docs, 1):
        logger.info(f"\nDocument {i}:")
        logger.info(f"  Metadata: {doc.metadata}")
        logger.info(f"  Content length: {len(doc.page_content)} characters")
    
    # Split documents
    docs_split = split_documents(docs)
    logger.info(f"\nSplit into {len(docs_split)} chunks")
    
    # Display chunk information (first 5 chunks)
    logger.info("\n" + "=" * 60)
    logger.info("Displaying First 5 Chunks")
    logger.info("=" * 60)
    
    for i, chunk in enumerate(docs_split[:5], 1):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"CHUNK {i}/{len(docs_split)}")
        logger.info(f"{'─' * 60}")
        
        # Display metadata
        if chunk.metadata:
            logger.info(f"Metadata: {chunk.metadata}")
        
        # Display content preview
        content = chunk.page_content
        preview_length = 200
        preview = content[:preview_length] + "..." if len(content) > preview_length else content
        logger.info(f"\nContent ({len(content)} chars):")
        logger.info(preview)
    
    # Ingest into vector store
    logger.info("\n" + "=" * 60)
    logger.info("Ingesting into Vector Store")
    logger.info("=" * 60)
    
    vectorstore = ingest_documents()
    logger.info(f"Ingested documents into: {settings.CHROMA_PERSIST_DIR}")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
