import streamlit as st
from pipeline import rag_pipeline, RAGState

# === Secrets ===
HF_TOKEN = st.secrets["HF_API_TOKEN"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# === App Config ===
st.set_page_config(page_title="10-K RAG Chatbot", layout="wide")
st.title("💬 10-K Company Insight Chatbot")
st.markdown("Ask strategic questions based on a company's 10-K filing.")

# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "company" not in st.session_state:
    st.session_state.company = ""

if "year" not in st.session_state:
    st.session_state.year = ""

# === Sidebar for Metadata ===
with st.sidebar:
    st.markdown("### 📋 Select Metadata")
    st.session_state.company = st.text_input("🏢 Company", st.session_state.company or "Workday")
    st.session_state.year = st.text_input("📅 Year", st.session_state.year or "2023")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# === Display Chat History ===
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Chat Input ===
user_input = st.chat_input("Ask a question about the 10-K report...")

if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 10-K data..."):

            # === Build RAG state ===
            rag_input: RAGState = {
                "query": user_input,
                "expanded_query": "",
                "company": st.session_state.company,
                "year": st.session_state.year,
                "context": None,
                "swot_analysis": None,
                "validated_response": "",
                "response_type": None,
                "final_answer": None,
                "confidence": ""
            }

            result = rag_pipeline.invoke(rag_input)

            # === Output response ===
            response = result["validated_response"]
            st.markdown(response)

            # Save assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            # Debug info
            with st.expander("🛠 Debug Info"):
                st.json(result)

