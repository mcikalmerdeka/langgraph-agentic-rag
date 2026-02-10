"""Gradio frontend for the LangGraph Agentic RAG application."""

import gradio as gr

from src.config import setup_logger, logger_frontend
from src.graph import rag_app

# Configure logging at module level
setup_logger(name="agentic_rag", level=20)  # 20 = INFO level

# Custom CSS for document display (passed to launch() in Gradio 6.0+)
CUSTOM_CSS = """
.documents-container details.doc-detail {
    margin: 10px 0 !important;
    border: 1px solid var(--border-color-primary, #e0e0e0) !important;
    border-radius: 8px !important;
    padding: 8px !important;
    background-color: var(--background-fill-secondary, #fafafa) !important;
}
.documents-container summary.doc-summary {
    cursor: pointer !important;
    font-weight: bold !important;
    color: var(--body-text-color, #1a1a1a) !important;
    padding: 4px !important;
}
.documents-container .doc-body {
    margin-top: 8px !important;
    padding: 8px !important;
    border-radius: 4px !important;
    background-color: var(--background-fill-primary, #ffffff) !important;
    color: var(--body-text-color, #1a1a1a) !important;
}
.documents-container .doc-body p {
    color: var(--body-text-color, #1a1a1a) !important;
    margin: 4px 0 !important;
}
.documents-container pre.doc-pre {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    background-color: var(--input-background-fill, #f0f0f0) !important;
    color: var(--body-text-color, #1a1a1a) !important;
    padding: 12px !important;
    border-radius: 4px !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
    border: 1px solid var(--border-color-primary, #e0e0e0) !important;
}
.documents-container p {
    color: var(--body-text-color, #1a1a1a) !important;
}
"""


def format_documents_html(documents):
    """Format documents as HTML with expandable accordions."""
    if not documents:
        return "<p>No documents retrieved</p>"
    
    html = f"<p><strong>Total documents retrieved: {len(documents)}</strong></p>\n"
    
    for i, doc in enumerate(documents, 1):
        source = doc.get("source", "unknown")
        title = doc.get("title", "N/A")
        content = doc.get("content", "")
        
        # Escape HTML special characters in content
        content_escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Format source for display (shorten if URL)
        source_display = source if len(source) <= 60 else source[:57] + "..."
        
        html += f"""
<details class="doc-detail">
    <summary class="doc-summary">📄 Document {i}: {source_display}</summary>
    <div class="doc-body">
        <p><strong>Title:</strong> {title[:80]}...</p>
        <p><strong>Source:</strong> {source}</p>
        <p><strong>Content:</strong></p>
        <pre class="doc-pre">{content_escaped}</pre>
    </div>
</details>
"""
    
    return html


def stream_response(question: str, search_type: str, k_documents: float, 
                   fetch_k: float, lambda_diversity: float, relevance_threshold: float,
                   max_web_results: float):
    """
    Stream RAG response with node-by-node updates and configurable retrieval settings.
    
    Args:
        question: The user's question
        search_type: Type of search ("similarity" or "mmr")
        k_documents: Number of documents to retrieve
        fetch_k: Number of candidates to fetch before selection (MMR only)
        lambda_diversity: Balance between relevance and diversity (0.0-1.0, MMR only)
        relevance_threshold: Minimum relevance score threshold
        
    Yields:
        Tuple of (chat_history, status_text, documents_html)
    """
    chat_history = []
    status_updates = []
    all_documents = []
    seen_document_hashes = set()
    
    # Add retrieval configuration to status
    config_msg = f"⚙️ Retrieval: {search_type.upper()} | k={k_documents}"
    if search_type == "mmr":
        config_msg += f" | fetch_k={fetch_k} | λ={lambda_diversity:.1f}"
    status_updates.append(config_msg)
    
    # Add user message
    chat_history.append({"role": "user", "content": question})
    yield chat_history, "\n".join(status_updates), "<p>No documents retrieved</p>"
    
    logger_frontend.info(f"User question received: {question}")
    logger_frontend.info(f"Retrieval config: type={search_type}, k={k_documents}, fetch_k={fetch_k}, lambda={lambda_diversity}")
    logger_frontend.info("Starting RAG processing pipeline")
    
    # Convert slider values to proper types and prepare retrieval config
    retrieval_config = {
        "search_type": search_type,
        "k": int(k_documents),
        "fetch_k": int(fetch_k),
        "lambda_mult": float(lambda_diversity),
        "score_threshold": float(relevance_threshold),
        "max_web_results": int(max_web_results)
    }
    
    # Log the exact configuration being used
    logger_frontend.info("=" * 60)
    logger_frontend.info("FRONTEND - Retrieval Configuration")
    logger_frontend.info("=" * 60)
    logger_frontend.info(f"Search type: {search_type}")
    logger_frontend.info(f"k (documents): {int(k_documents)}")
    logger_frontend.info(f"fetch_k: {int(fetch_k)}")
    logger_frontend.info(f"lambda: {float(lambda_diversity)}")
    logger_frontend.info(f"threshold: {float(relevance_threshold)}")
    logger_frontend.info(f"max_web_results: {int(max_web_results)}")
    logger_frontend.info("=" * 60)
    
    # Stream through the graph
    full_response = ""
    for chunk in rag_app.stream(input={"question": question, "retrieval_config": retrieval_config}):
        for node, update in chunk.items():
            status_msg = f"▶ Processing node: {node}"
            status_updates.append(status_msg)
            logger_frontend.info(f"Processing node: {node}")
            
            # Handle documents
            if "documents" in update:
                docs = update["documents"]
                if docs:
                    logger_frontend.info(f"Retrieved {len(docs)} documents from {node}")
                    # Store detailed document information with deduplication
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "unknown")
                        title = doc.metadata.get("title", "N/A")
                        content = doc.page_content
                        
                        # Create hash for deduplication (based on content + source)
                        doc_hash = hash((content[:200], source))
                        
                        if doc_hash not in seen_document_hashes:
                            seen_document_hashes.add(doc_hash)
                            
                            doc_data = {
                                "id": f"doc_{len(all_documents) + 1}",
                                "source": source,
                                "title": title,
                                "content": content,
                                "metadata": doc.metadata
                            }
                            all_documents.append(doc_data)
                            logger_frontend.debug(f"Added document {len(all_documents)}: {source}")
            
            # Handle web search flag
            if "web_search" in update:
                if update["web_search"]:
                    status_msg = "🔍 Web search enabled"
                    status_updates.append(status_msg)
                    logger_frontend.info("Web search enabled")
                else:
                    status_msg = "📚 Using vector store only"
                    status_updates.append(status_msg)
                    logger_frontend.info("Using vector store only")
            
            # Handle generation
            if "generation" in update:
                full_response = update["generation"]
                # Update chat history with assistant response
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = full_response
                else:
                    chat_history.append({"role": "assistant", "content": full_response})
                logger_frontend.info("Response generated successfully")
            
            # Yield current state
            status_text = "\n".join(status_updates[-10:])
            docs_html = format_documents_html(all_documents)
            yield chat_history, status_text, docs_html
    
    logger_frontend.info("RAG processing pipeline complete")
    # Final yield with complete status
    status_text = "\n".join(status_updates) if status_updates else "✅ Complete"
    docs_html = format_documents_html(all_documents)
    yield chat_history, status_text, docs_html
    
    # Final yield with complete status
    status_text = "\n".join(status_updates) if status_updates else "✅ Complete"
    docs_html = format_documents_html(all_documents)
    yield chat_history, status_text, docs_html


def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface with retrieval settings sidebar."""
    with gr.Blocks(title="LangGraph Agentic RAG") as app:
        gr.Markdown("# LangGraph Agentic RAG")
        gr.Markdown("Ask questions about AI concepts. The system will automatically decide whether to use the knowledge base or web search.")
        
        with gr.Sidebar(label="⚙️ Retrieval Settings") as sidebar:
            gr.Markdown("### Configure Document Retrieval")
            
            # Search type selection
            search_type = gr.Dropdown(
                choices=["mmr", "similarity"],
                value="mmr",
                label="Search Method",
                info="MMR: Diverse results | Similarity: Top matches only"
            )
            
            # Number of documents to retrieve
            k_documents = gr.Slider(
                minimum=1,
                maximum=20,
                value=6,
                step=1,
                label="Documents to Retrieve (k)",
                info="Number of final documents to return"
            )
            
            # Relevance threshold (above MMR settings)
            relevance_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Minimum Relevance Score",
                info="Documents below this score will be filtered out"
            )
            
            # MMR-specific settings (in an accordion)
            with gr.Accordion("MMR Advanced Settings", open=False):
                fetch_k = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Fetch Candidates (fetch_k)",
                    info="Number of candidates to fetch before MMR selection"
                )
                
                lambda_diversity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Diversity Balance (λ)",
                    info="0.0 = Max diversity, 1.0 = Max relevance"
                )
            
            # Web search settings
            with gr.Accordion("Web Search Settings", open=False):
                max_web_results = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Max Web Search Results",
                    info="Number of web results to fetch when needed"
                )
            
            # Reset button
            reset_btn = gr.Button("Reset to Defaults", variant="secondary")
            
            def reset_settings():
                return ["mmr", 6, 20, 0.5, 0.3, 2]
            
            reset_btn.click(
                fn=reset_settings,
                outputs=[search_type, k_documents, fetch_k, lambda_diversity, relevance_threshold, max_web_results]
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation"
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Enter your question here...",
                        label="Question",
                        scale=4
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                
                # Example questions: knowledge base vs web search
                gr.Markdown("**Example Questions**")
                gr.Examples(
                    examples=[
                        "What is agent memory?",
                        "Can you explain concept of few-shot prompting?",
                    ],
                    inputs=question_input,
                    label="From knowledge base"
                )
                gr.Examples(
                    examples=[
                        "What is definition of Context Engineering and when did it get popular?",
                        "What are best places to visit in Indonesia?",
                    ],
                    inputs=question_input,
                    label="Require web search"
                )
                
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                # Status panel
                gr.Markdown("### Processing Status")
                status_text = gr.Textbox(
                    label="Node Updates",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
                
                gr.Markdown("### Retrieved Documents")
                documents_html = gr.Markdown(
                    label="Document Details",
                    value="No documents retrieved",
                    elem_classes=["documents-container"]
                )
                gr.Markdown("*Click on any document to expand and see full content*")
        
        # Event handlers
        submit_btn.click(
            fn=stream_response,
            inputs=[question_input, search_type, k_documents, fetch_k, lambda_diversity, relevance_threshold, max_web_results],
            outputs=[chatbot, status_text, documents_html]
        )
        
        question_input.submit(
            fn=stream_response,
            inputs=[question_input, search_type, k_documents, fetch_k, lambda_diversity, relevance_threshold, max_web_results],
            outputs=[chatbot, status_text, documents_html]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", "No documents retrieved"),
            outputs=[chatbot, status_text, documents_html]
        )
    
    return app


# Create the interface instance
app = create_interface()

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(), share=False, css=CUSTOM_CSS)
