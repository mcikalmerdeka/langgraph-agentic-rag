import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represent the state of the RAG graph.

    Attributes:
        question: The user's question
        generation: The LLM's generated response
        web_search: Flag indicating whether to perform web search
        documents: List of retrieved/relevant documents
        retrieval_config: Configuration for document retrieval (search type, k, etc.)
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[List[Document], operator.add]
    retrieval_config: Dict[str, Any]

