"""Main entry point for the LangGraph Agentic RAG application."""

import gradio as gr

from src.core import setup_logging
from src.frontend import app

# Configure logging level: DEBUG, INFO, WARNING, ERROR
setup_logging("INFO")


def main() -> None:
    """Launch the Gradio frontend for the RAG application."""
    print("=" * 60)
    print("LangGraph Agentic RAG Application")
    print("=" * 60)
    print("\nStarting Gradio interface...")
    print("Open your browser to the displayed URL\n")
    
    # Launch the Gradio app with theme
    app.launch(theme=gr.themes.Soft(), share=False)


if __name__ == "__main__":
    main()
