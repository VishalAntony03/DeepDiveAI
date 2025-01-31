import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# Load models
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key="-proj-qsi_Ixu3fbZc5o4SDML5WIxTAM8to7hvYXOKGoM8RTgTiABRLZRTsuc5fjyVi6ytLJcLEQ_IHET3BlbkFJ7O5p90huhaOx1sOCC7NNb4kfXJ5qh0EuC1d6dWxeHSTXJfcvIhZBzF_ZW94D3n0-PvLqs3OskA")  # Replace with your API key

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def create_faiss_index(sentences):
    """Creates a FAISS index for fast similarity search."""
    vectors = encoder.encode(sentences, convert_to_numpy=True)
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index, vectors, sentences

def retrieve_relevant_sections(query, index, vectors, sentences, k=3):
    """Retrieves the most relevant sections for a given query."""
    query_vector = encoder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vector, k)
    return [sentences[i] for i in indices[0] if i < len(sentences)]

def generate_response(context, query):
    """Generates an AI-powered response based on retrieved context."""
    prompt = f"""You are an AI assistant. Using the following context, answer the question:
    Context: {context}
    Question: {query}
    Answer:"""
    response = client.completions.create(model="gpt-3.5-turbo", prompt=prompt, max_tokens=200)
    return response.choices[0].text.strip()

# Streamlit UI
st.set_page_config(page_title="AI Research Paper Q&A", layout="wide")
st.title("📄 AI Research Paper Q&A")
st.markdown("Upload a research paper and ask questions to get insights!")

uploaded_file = st.file_uploader("Upload an AI Research Paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        sentences = text.split(". ")
        index, vectors, sentences = create_faiss_index(sentences)
    st.success("PDF successfully processed and indexed! ✅")

    query = st.text_input("🔍 Ask a question about the paper:")
    if st.button("Get Answer", use_container_width=True) and query:
        with st.spinner("Retrieving answer..."):
            relevant_sections = retrieve_relevant_sections(query, index, vectors, sentences)
            context = " ".join(relevant_sections)
            answer = generate_response(context, query)
        st.subheader("💡 Answer:")
        st.write(answer)