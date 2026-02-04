"""Gradio frontend for the LangGraph Agentic RAG application."""

import gradio as gr

from src.core import setup_logging
from src.graph import rag_app

# Configure logging
setup_logging("INFO")


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


def stream_response(question: str):
    """
    Stream RAG response with node-by-node updates.
    
    Args:
        question: The user's question
        
    Yields:
        Tuple of (chat_history, status_text, documents_html)
    """
    chat_history = []
    status_updates = []
    all_documents = []
    seen_document_hashes = set()
    
    # Add user message
    chat_history.append({"role": "user", "content": question})
    yield chat_history, "Processing...", "<p>No documents retrieved</p>"
    
    # Stream through the graph
    full_response = ""
    for chunk in rag_app.stream(input={"question": question}):
        for node, update in chunk.items():
            status_updates.append(f"Processing node: {node}")
            
            # Handle documents
            if "documents" in update:
                docs = update["documents"]
                if docs:
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
            
            # Handle web search flag
            if "web_search" in update:
                if update["web_search"]:
                    status_updates.append("Web search enabled")
                else:
                    status_updates.append("Using vector store only")
            
            # Handle generation
            if "generation" in update:
                full_response = update["generation"]
                # Update chat history with assistant response
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = full_response
                else:
                    chat_history.append({"role": "assistant", "content": full_response})
            
            # Yield current state
            status_text = "\n".join(status_updates[-10:])
            docs_html = format_documents_html(all_documents)
            yield chat_history, status_text, docs_html
    
    # Final yield with complete status
    status_text = "\n".join(status_updates) if status_updates else "Complete"
    docs_html = format_documents_html(all_documents)
    yield chat_history, status_text, docs_html
    
    # Final yield with complete status
    status_text = "\n".join(status_updates) if status_updates else "Complete"
    docs_html = format_documents_html(all_documents)
    yield chat_history, status_text, docs_html


def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    custom_css = """
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
    with gr.Blocks(title="LangGraph Agentic RAG", css=custom_css) as app:
        gr.Markdown("# LangGraph Agentic RAG")
        gr.Markdown("Ask questions about AI concepts. The system will automatically decide whether to use the knowledge base or web search.")
        
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
                    lines=8,
                    max_lines=12,
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
            inputs=question_input,
            outputs=[chatbot, status_text, documents_html]
        )
        
        question_input.submit(
            fn=stream_response,
            inputs=question_input,
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
    app.launch(theme=gr.themes.Soft(), share=False)
