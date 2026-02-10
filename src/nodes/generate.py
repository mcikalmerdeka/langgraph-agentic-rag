"""Generate node - produces answers using the generation chain."""

from typing import Any, Dict, List

from langchain_core.documents import Document

from src.chains import generation_chain
from src.core import logger
from src.core.state import GraphState


def format_documents_for_context(documents: List[Document]) -> str:
    """
    Format documents into a readable context string with sources.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "N/A")
        content = doc.page_content.strip()
        
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"{'='*60}"
        )
    
    return "\n\n".join(formatted)


def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer using the retrieved documents.

    Args:
        state: Current graph state with question and documents.

    Returns:
        Updated state with generation.
    """
    logger.debug("Generating answer...")

    question = state["question"]
    documents = state["documents"]
    
    # Format documents with sources for better context
    context = format_documents_for_context(documents)
    logger.debug(f"Formatted {len(documents)} documents into context")

    generation = generation_chain.invoke({"question": question, "context": context})

    logger.info(f"Generated response ({len(generation)} chars)")

    return {"generation": generation, "documents": documents}
