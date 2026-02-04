"""Vector store ingestion and retrieval logic."""

from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.core import logger
from src.core.llm import get_embeddings

# Default URLs for the knowledge base
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def load_documents(urls: List[str] | None = None) -> List[Document]:
    """Load documents from URLs with proper metadata."""
    urls = urls or DEFAULT_URLS
    all_docs = []
    
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Ensure source metadata is set for each document
        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = url
        
        all_docs.extend(docs)
    
    return all_docs


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_documents(urls: List[str] | None = None) -> Chroma:
    """
    Load, split, and store documents in ChromaDB.
    
    Args:
        urls: List of URLs to ingest. Uses DEFAULT_URLS if not provided.
        
    Returns:
        The ChromaDB vector store instance.
    """
    docs = load_documents(urls)
    logger.info(f"Loaded {len(docs)} documents from URLs")
    
    # Log metadata info before splitting
    for i, doc in enumerate(docs[:3], 1):  # Log first 3 docs
        logger.info(f"Doc {i} metadata: {doc.metadata}")
    
    docs_split = split_documents(docs)
    logger.info(f"Split into {len(docs_split)} chunks")

    # Verify metadata is preserved after splitting
    for i, doc in enumerate(docs_split[:3], 1):  # Log first 3 chunks
        logger.info(f"Chunk {i} metadata: {doc.metadata}")

    vectorstore = Chroma.from_documents(
        documents=docs_split,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vectorstore


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """
    Get the retriever for similarity search (cached).
    
    Assumes documents have already been ingested.
    """
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vectorstore.as_retriever()

