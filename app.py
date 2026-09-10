import gradio as gr
from rag_pipeline import RAGPipeline


rag = RAGPipeline()


def chat_with_rag(message, history):
    if not message.strip():
        return "Please enter a question."

    answer, sources = rag.answer_question(message, top_k=5)

    return f"""
{answer}

---

### Retrieved News Articles

{sources}
"""


demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="Enterprise Document Intelligence Assistant",
    description=(
        "Ask questions about the BBC News document collection. "
        "The assistant retrieves relevant document chunks using FAISS "
        "and generates grounded answers using an LLM."
    ),
    examples=[
        "What is happening in the economy and financial markets?",
        "What is happening in sports news?",
        "What are the latest updates in politics?",
        "What technology news is discussed?",
        "What entertainment news is available?",
    ],
)


if __name__ == "__main__":
    try:
        import google.colab  # noqa: F401
        running_in_colab = True
    except ImportError:
        running_in_colab = False

    demo.launch(share=running_in_colab)