"""Main entry point for the LangGraph Agentic RAG application."""

import gradio as gr

from src.config import setup_logger, logger_frontend
from src.frontend import app, CUSTOM_CSS

# Configure logging at startup - logs to terminal and logs/app.log
setup_logger(name="agentic_rag", level=20)  # 20 = INFO level


def main() -> None:
    """Launch the Gradio frontend for the RAG application."""
    logger_frontend.info("=" * 60)
    logger_frontend.info("LangGraph Agentic RAG Application")
    logger_frontend.info("=" * 60)
    logger_frontend.info("Starting Gradio interface...")
    logger_frontend.info("Open your browser to the displayed URL")
    logger_frontend.info("Retrieval settings available in the sidebar")
    
    # Launch the Gradio app with theme and CSS (Gradio 6.0+ requirement)
    app.launch(theme=gr.themes.Soft(), share=False, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
