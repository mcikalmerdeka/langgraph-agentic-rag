"""Retrieve node - fetches relevant documents from vector store with configurable settings."""

from typing import Any, Dict

from src.core import logger
from src.core.state import GraphState
from src.ingestion import get_retriever


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents for the question with configurable retrieval settings.

    Args:
        state: Current graph state with question and optional retrieval_config.

    Returns:
        Updated state with retrieved documents.
    """
    question = state["question"]
    
    # Get retrieval configuration from state or use defaults
    retrieval_config = state.get("retrieval_config", {})
    search_type = retrieval_config.get("search_type", "mmr")
    k = retrieval_config.get("k", 6)
    fetch_k = retrieval_config.get("fetch_k", 20)
    lambda_mult = retrieval_config.get("lambda_mult", 0.5)
    score_threshold = retrieval_config.get("score_threshold", 0.3)
    
    logger.info("=" * 60)
    logger.info("RETRIEVE NODE - Configuration")
    logger.info("=" * 60)
    logger.info(f"Search type: {search_type}")
    logger.info(f"k (documents to retrieve): {k}")
    logger.info(f"fetch_k (candidates): {fetch_k}")
    logger.info(f"lambda_mult (diversity): {lambda_mult}")
    logger.info(f"score_threshold: {score_threshold}")
    logger.info(f"Question: {question[:50]}...")
    logger.info("=" * 60)
    
    # Get retriever with specified configuration
    retriever = get_retriever(
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        score_threshold=score_threshold
    )
    
    documents = retriever.invoke(question)
    
    logger.info(f"✓ Retrieved {len(documents)} documents (requested k={k})")
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        logger.info(f"  Doc {i}: {source[:60]}...")
    
    return {"documents": documents}
