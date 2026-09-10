import streamlit as st

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Enterprise Document Intelligence Assistant", page_icon="📰")


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


rag = load_pipeline()

st.title("Enterprise Document Intelligence Assistant")
st.caption(
    "Ask questions about the BBC News document collection. The assistant retrieves "
    "relevant document chunks using FAISS and generates grounded answers using an LLM."
)

EXAMPLES = [
    "What is happening in the economy and financial markets?",
    "What is happening in sports news?",
    "What are the latest updates in politics?",
    "What technology news is discussed?",
    "What entertainment news is available?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Example questions")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_question = st.chat_input("Ask a question about the BBC News collection...")
question = st.session_state.pop("pending_question", None) or typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving articles and generating an answer..."):
            answer, sources = rag.answer_question(question, top_k=5)
            response = f"{answer}\n\n---\n\n**Retrieved News Articles**\n\n{sources}"
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
