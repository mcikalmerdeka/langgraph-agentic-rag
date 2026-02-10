"""Vector store ingestion and retrieval logic."""

import os
from functools import lru_cache
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.config import logger_ingestion as logger
from src.core.llm import get_embeddings

# Set USER_AGENT environment variable to avoid warnings
os.environ.setdefault("USER_AGENT", "LangGraph-Agentic-RAG/1.0")

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


def split_documents(
    docs: List[Document], 
    chunk_size: int = 1000,  # Increased from 500 for better context
    chunk_overlap: int = 200  # Increased overlap to maintain continuity
) -> List[Document]:
    """Split documents into chunks with optimal settings for RAG."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize paragraph breaks
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


def get_retriever(
    search_type: str = "mmr",
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: float = 0.3,
    vectorstore: Optional[Chroma] = None
) -> VectorStoreRetriever:
    """
    Get the retriever for similarity search with configurable settings.
    
    Args:
        search_type: "mmr" for diverse results, "similarity" for top-k similar
        k: Number of documents to retrieve (default: 6)
        fetch_k: Number of candidates to fetch before MMR selection (default: 20)
        lambda_mult: Balance between relevance (1.0) and diversity (0.0) for MMR (default: 0.5)
        score_threshold: Minimum relevance score threshold (default: 0.3)
        vectorstore: Optional pre-initialized vectorstore. If None, creates new connection.
    
    Returns:
        Configured VectorStoreRetriever instance
    """
    if vectorstore is None:
        vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    
    if search_type == "mmr":
        # MMR provides diverse results
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,  # Fetch more candidates for diversity
                "lambda_mult": lambda_mult  # Balance between relevance and diversity
            }
        )
        logger.info(f"Created MMR retriever: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}")
    elif search_type == "similarity_score_threshold":
        # Similarity search with score threshold
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )
        logger.info(f"Created similarity threshold retriever: k={k}, threshold={score_threshold}")
    else:
        # Standard similarity search
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        logger.info(f"Created similarity retriever: k={k}")
    
    return retriever

