import streamlit as st
import ollama
from pypdf import PdfReader

st.title("⚖️ AI Legal Document Assistant")

pdf = st.file_uploader("Upload a legal PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    st.success("Document uploaded successfully!")

    question = st.text_input("Ask a question about the document:")

    if question:
        prompt = f"""
        Based only on the following legal document, answer the question.
        If the answer is not found, say "Not found in the document."

        Document:
        {text}

        Question:
        {question}
        """

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response["message"]["content"])