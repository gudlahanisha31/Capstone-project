import streamlit as st
import chromadb
import ollama
import tempfile
import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Legal Document Assistant",
    page_icon="⚖️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚖️ AI Legal Document Assistant")
st.write(
    "Upload a legal document and ask questions about its contents using AI."
)

st.warning(
    "⚠️ This tool provides information from the uploaded document "
    "and is not a substitute for professional legal advice."
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

@st.cache_resource
def create_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="legal_documents"
    )

    return collection


collection = create_database()


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# ADD DOCUMENT TO DATABASE
# --------------------------------------------------

def add_document_to_database(chunks):

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


# --------------------------------------------------
# SEARCH DOCUMENT
# --------------------------------------------------

def search_document(question, number_of_results=4):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])[0]

    return documents


# --------------------------------------------------
# OLLAMA ANSWER
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are an AI legal document assistant.

Answer the user's question ONLY using the information
provided in the document context below.

If the answer is not present in the document, say:

"I could not find this information in the uploaded document."

Do not invent laws, clauses, dates, names, penalties,
or legal conclusions.

Explain the answer in simple language.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📄 Upload Legal Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file:

    if st.sidebar.button("Process Document"):

        with st.spinner("Reading legal document..."):

            try:

                text = extract_text_from_pdf(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "No readable text was found in this PDF."
                    )

                else:

                    chunks = split_text(text)

                    # Create unique IDs for this upload
                    ids = [
                        f"{uploaded_file.name}_{i}"
                        for i in range(len(chunks))
                    ]

                    embeddings = embedding_model.encode(
                        chunks
                    ).tolist()

                    collection.add(
                        ids=ids,
                        documents=chunks,
                        embeddings=embeddings
                    )

                    st.success(
                        f"Document processed successfully! "
                        f"{len(chunks)} text chunks stored."
                    )

                    st.session_state["document_uploaded"] = True

            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )


# --------------------------------------------------
# DOCUMENT INFORMATION
# --------------------------------------------------

if uploaded_file:

    st.subheader("📄 Uploaded Document")

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Size:** {uploaded_file.size / 1024:.2f} KB"
    )


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions About Your Legal Document")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What are the termination conditions?"
)


# --------------------------------------------------
# ASK BUTTON
# --------------------------------------------------

if st.button("🔍 Ask AI"):

    if not uploaded_file:

        st.error(
            "Please upload and process a PDF first."
        )

    elif not question.strip():

        st.error(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                relevant_documents = search_document(
                    question
                )

                if not relevant_documents:

                    st.warning(
                        "No relevant information found."
                    )

                else:

                    context = "\n\n".join(
                        relevant_documents
                    )

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = generate_answer(
                            question,
                            context
                        )

                    st.subheader("🤖 AI Answer")

                    st.write(answer)

                    # Show supporting context
                    with st.expander(
                        "📚 View Relevant Document Sections"
                    ):

                        for i, document in enumerate(
                            relevant_documents,
                            start=1
                        ):

                            st.markdown(
                                f"**Section {i}**"
                            )

                            st.write(document)

                            st.divider()

            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )


# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

st.divider()

st.subheader("🎤 Voice Question")

audio_file = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "mp3", "m4a"]
)


@st.cache_resource
def load_whisper():

    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    return model


if audio_file:

    if st.button("🎤 Convert Voice to Text"):

        with st.spinner(
            "Converting speech to text..."
        ):

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp:

                    temp.write(
                        audio_file.read()
                    )

                    audio_path = temp.name

                whisper_model = load_whisper()

                segments, info = whisper_model.transcribe(
                    audio_path
                )

                transcribed_text = " ".join(
                    segment.text
                    for segment in segments
                )

                st.success(
                    "Voice converted successfully!"
                )

                st.write(
                    f"**Your question:** {transcribed_text}"
                )

                os.remove(audio_path)

            except Exception as e:

                st.error(
                    f"Voice processing error: {e}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Legal Document Assistant | "
    "Streamlit + ChromaDB + Sentence Transformers + Ollama"
)