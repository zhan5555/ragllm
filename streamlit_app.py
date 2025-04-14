import streamlit as st
import os
from your_rag_module import rag_pipeline  # replace with your actual module
from your_rag_module import RAGState  # import the schema if needed

# Load from secrets
HF_TOKEN = st.secrets["HF_API_TOKEN"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

st.set_page_config(page_title="10-K LLM RAG", layout="wide")
st.title("🔍 10-K LLM Assistant")
st.markdown("Ask questions about a company's strategy based on their 10-K filings.")

# User input form
with st.form(key="query_form"):
    query = st.text_input("💬 Your question", placeholder="e.g., What are the risks to Workday’s AI product strategy?")
    company = st.text_input("🏢 Company name (optional)", placeholder="e.g., Workday")
    year = st.text_input("📅 Filing year (optional)", placeholder="e.g., 2023")
    submit = st.form_submit_button("Run Analysis")

if submit:
    with st.spinner("Running RAG pipeline..."):
        initial_state: RAGState = {
            "query": query,
            "expanded_query": "",
            "company": company,
            "year": year,
            "context": None,
            "swot_analysis": None,
            "validated_response": "",
            "response_type": None,
            "final_answer": None,
            "confidence": ""
        }

        result = rag_pipeline.invoke(initial_state)
        st.subheader("📄 Validated Answer")
        st.write(result["validated_response"])
        st.caption(f"Confidence: {result['confidence']}")

        with st.expander("🧠 Debug Info"):
            st.json(result)
