import os
from langgraph.graph import StateGraph, END
from typing import TypedDict
from google.generativeai import GenerativeModel
from prompts import PromptTemplates
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from transformers import pipeline

# === Gemini setup ===
gemini = GenerativeModel("gemini-2.0-flash")

# SECTION 3: Authentication and PINECONE SETUP
# huuggingface_hub login
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Pinecone set up
EMBEDDING_DIM = 3584
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "documentvectorstore")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

def init_pinecone_index(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-1")
        )
    return pc.Index(index_name)

# Initialize at module level
index = init_pinecone_index(PINECONE_API_KEY, INDEX_NAME)

# ✅ SECTION 4: EMBEDDING PIPELINE (Qwen2-7B with Mean Pooling)
embed_pipe = pipeline(
    "feature-extraction",
    model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    token=HF_API_TOKEN,
    device=-1  # CPU
)

def embed_text_batch(texts: list) -> np.ndarray:
    """
    Embeds a list of texts using Qwen2-7B with mean pooling.
    Ensures output shape is (batch_size, 3584).
    """
    if isinstance(texts, str):
        texts = [texts]

    outputs = embed_pipe(texts, padding=True, truncation=True, max_length=512)
    normalized_outputs = []
    for out in outputs:
        arr = np.array(out)
        while arr.ndim > 2:
            arr = arr[0]  # unwrap unnecessary batch dims
        normalized_outputs.append(np.mean(arr, axis=0))  # mean pooling

    return np.vstack(normalized_outputs)

def embed_text(text):
    return embed_text_batch([text])[0].tolist()  # ← ensures correct format

# === LangGraph State ===
class RAGState(TypedDict):
    query: str
    expanded_query: str
    response_type: str
    retrieved_docs: str
    final_answer: str
    debug_info: str              # ✅ NEW: for LLM validation output
    references: list             # ✅ NEW: for chunk metadata like page number & PDF
    company: str                  # ✅ for filtering chunks with specific company
    year: str                     # ✅ for filtering chunks with specific company's 10K year

# === Debug Control ===
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

def debug_print(label, value):
    if DEBUG_MODE:
        print(f"\n🔍 {label}:\n{value}\n")

# === LangGraph Nodes ===

def expand_query_node(state: RAGState) -> RAGState:
    prompt = PromptTemplates.EXPAND_QUERY.format(query=state["query"])
    response = gemini.generate_content(prompt).text.strip()
    state["expanded_query"] = response
    debug_print("Expanded Query", response)
    return state

def classify_query_node(state: RAGState) -> RAGState:
    prompt = PromptTemplates.CLASSIFY_QUERY.format(query=state["expanded_query"])
    response = gemini.generate_content(prompt).text.strip()
    state["response_type"] = response.split("\n")[0]
    debug_print("Response Type", state["response_type"])
    return state

def retrieve_docs_node(state: RAGState) -> RAGState:
    query_embedding = embed_text(state["expanded_query"])
    filter_criteria = {
        "company": state.get("company"),
        "year": state.get("year")
    }
    filter_criteria = {k: v for k, v in filter_criteria.items() if v}  # remove None

    search_result = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter=filter_criteria       # ✅ NEW
    )

    # 🔍 Debug: how many matches returned
    debug_print("Pinecone Match Count", len(search_result["matches"]))

    chunks = []
    references = []
    for match in search_result["matches"]:
        chunk = match["metadata"].get("chunk_text")
        if chunk:
            chunks.append(chunk)
            references.append({
                "page": match["metadata"].get("page_num"),
                "pdf_path": match["metadata"].get("pdf_path")
            })

    docs = "\n\n".join(chunks)
    state["retrieved_docs"] = docs
    state["references"] = references  # ✅ used in Streamlit for citation

    debug_print("Retrieved Docs", docs)         # optional for visibility
    debug_print("Retrieved References", references)  # optional for traceability

    return state 

def generate_response_node(state: RAGState) -> RAGState:
    query = state["query"]
    docs = state["retrieved_docs"]
    prompt_template = {
        "metric_snapshot": PromptTemplates.METRIC_SNAPSHOT,
        "cot_response": PromptTemplates.CHAIN_OF_THOUGHT,
        "swot_analysis": PromptTemplates.SWOT_ANALYSIS,
    }.get(state["response_type"], PromptTemplates.CHAIN_OF_THOUGHT)

    prompt = prompt_template.format(query=query)
    full_input = f"{prompt}\n\nContext:\n{docs}"
    response = gemini.generate_content(full_input).text.strip()
    state["final_answer"] = response
    debug_print("Generated Answer", response)
    return state

def validate_response_node(state: RAGState) -> RAGState:
    prompt = PromptTemplates.VALIDATE_RESPONSE.format(
        query=state["query"],
        response=state["final_answer"],
        context=state["retrieved_docs"]
    )
    validation = gemini.generate_content(prompt).text.strip()
    state["debug_info"] = validation      # ✅ NEW: store model validation response
    debug_print("Validation Result", validation)
    return state

# === LangGraph Pipeline ===
workflow = StateGraph(RAGState)
workflow.add_node("expand_query", expand_query_node)
workflow.add_node("classify_query", classify_query_node)
workflow.add_node("retrieve_docs", retrieve_docs_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("validate_response", validate_response_node)

workflow.set_entry_point("expand_query")
workflow.add_edge("expand_query", "classify_query")
workflow.add_edge("classify_query", "retrieve_docs")
workflow.add_edge("retrieve_docs", "generate_response")
workflow.add_edge("generate_response", "validate_response")
workflow.add_edge("validate_response", END)

rag_pipeline = workflow.compile()

