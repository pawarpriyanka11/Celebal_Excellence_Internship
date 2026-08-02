import os
import streamlit as st
from rag_pipeline import DocumentRAG


st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")

st.title("📄 Document Question Answering with RAG")
st.write("Upload a PDF or text file, then ask questions about it.")

if "rag" not in st.session_state:
    st.session_state.rag = None

uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    paths = []
    for uploaded_file in uploaded_files:
        save_path = os.path.join(temp_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(save_path)

    with st.spinner("Building the retrieval index..."):
        rag = DocumentRAG()
        rag.load_documents(paths)
        st.session_state.rag = rag

    st.success("Document index created successfully.")
elif st.session_state.rag is None:
    default_docs = []
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.isdir(data_dir):
        for file_name in sorted(os.listdir(data_dir)):
            if file_name.lower().endswith((".txt", ".pdf")):
                default_docs.append(os.path.join(data_dir, file_name))

    if default_docs:
        with st.spinner("Loading the sample document..."):
            rag = DocumentRAG()
            rag.load_documents(default_docs)
            st.session_state.rag = rag
        st.info("Loaded the sample document from the data folder.")

query = st.text_input("Ask a question about your document")

if st.button("Generate Answer") and query:
    if st.session_state.rag is None:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Retrieving relevant context..."):
            answer = st.session_state.rag.answer(query)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Context")
        for i, (chunk, distance) in enumerate(st.session_state.rag.retrieve(query), 1):
            st.write(f"{i}. {chunk}")
            st.caption(f"Distance: {distance:.4f}")
