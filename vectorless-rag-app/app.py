import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag import VectorlessRAG

WORKSPACE_DIR = "./workspace"

def init_session_state():
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def initialize_rag():
    if st.session_state.rag is None:
        try:
            st.session_state.rag = VectorlessRAG(workspace_dir=WORKSPACE_DIR)
        except ValueError as e:
            st.error(f"Error initializing RAG system: {e}")
            st.stop()


def get_documentt_list():
    if st.session_state.rag is None:
        return []
    return st.session_state.rag.list_documents()

def index_document(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_path = tmp_file.name

    try:
        for doc in st.session_state.rag.list_documents():
            if doc.get('doc_name') == upload_file.name:
                for doc_id in st.session_state.rag.client.documents.item():
                    if d.get('doc_name')== upload_file.name:
                        return doc_id, "Document already indexed"
        doc_id = st.session_state.rag.index_document(tmp_path)
        return doc_id, None
    except Exception as e:
        return None, str(e)
    finally:
        os.remove(tmp_path)


def query_document(doc_id, question: str) -> str:
    return st.session_state.rag.query(doc_id, question)

def main():
    st.set_page_config(page_title="Vectorless RAG App", layout="wide", page_icon="📄")

    init_session_state()
    initialize_rag()

    st.title("Vectorless RAG App")
    st.markdown("Upload a PDF document to index it, then ask questions about its content.")

    with st.sidebar:
        st.header("Upload Document")
        st.subheader("Step 1: Upload a PDF file")
        upload_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"],
             help="Upload a PDF document to index its content for question-answering."
        )
        if upload_file is not None:
            if st.button("Index Document", type="primary"):
                with st.spinner(f"Indexing {upload_file.name}...This may take a moment."):
                    doc_id, error = index_document(upload_file)
                    if error:
                        if "already indexed" in error.lower():
                            st.success(f" {error}")
                            st.session_state.current_doc_id = doc_id
                        else:
                            st.error(f"Error indexing document: {error}")
                    else:
                        st.success(f"Document indexed successfully with ID: {doc_id}")
                        st.session_state.current_doc_id = doc_id
                        st.session_state.chat_history = []  # Clear chat history when a new document is indexed
                        st.rerun()  # Refresh the app to show the new document in the sidebar

        st.divider()

        st.subheader("select Document")
        docs = get_documentt_list()

        if docs:
            doc_ids= list(st.session_state.rag.client.documents.keys())
            doc_options = {}
            for i, doc in enumerate(docs):
                name = doc.get('doc_name', 'unknown')
                pages = doc.get('page_count',0)
                doc_options[f"{name} ({pages} pages)"] = doc_ids[i]
            
            selected = st.selectbox(
                "choose a document to query",
                options=list(doc_options.keys()),
                index=0 if st.session_state.current_doc_id is None else 
                      list(doc_options.values()).index(st.session_state.current_doc_id)
                      if st.session_state.current_doc_id in doc_options.values() else 0

            )

            if selected:
                new_doc_id = doc_options[selected]
                if new_doc_id != st.session_state.current_doc_id:
                    st.session_state.current_doc_id = new_doc_id
                    st.session_state.chat_history = []  # Clear chat history when switching documents
                    
        else:
            st.info("No documents indexed yet. Please upload a PDF to get started.")

        st.divider()

        st.subheader("Configuration")
        if st.session_state.rag:
            st.write(f"**model:** {st.session_state.rag.model}")
        st.write(f"**workspace** {WORKSPACE_DIR}")
    
    if st.session_state.current_doc_id:
        doc_info = st.session_state.rag.get_document_info(st.session_state.current_doc_id)

        tab1, tab2, tab3 = st.tabs(["Chat", "Document Info", "Document Structure"])

        with tab1:
            st.subheader(f"chat with: {doc_info.get('doc_name', 'Document')}")

            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(msg['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg['content'])
                        if msg.get('citation'):
                            st.markdown(f"**Citation:** {msg['citation']}")
                        if msg.get('confidence'):
                            confidence_colour = {
                                'high': 'green',
                                'medium': 'orange',
                                'low': 'red'
                            }.get(msg['confidence'], 'black')
                            st.caption(f"{confidence_colour}Confidence: {msg['confidence']}")
                            
        question = st.chat_input("Ask a question about the document...")     

        if question:
            st.session_state.chat_history.append({'role': 'user', 'content': question})

            with st.chat_message("user"):
                st.write(question)
            
            with st.spinner("Assistant"):
                with st.spinner("Reasoning"):
                    result = query_document(st.session_state.current_doc_id, question)

                st.write(result.get('answer', 'No answer returned'))

                citation = result.get('citation', [])
                if citation:
                    st.caption(f"**Citation:** {citation}")

                confidence = result.get('confidence', 'unknown')
                confidence_colour = {   
                    'high': 'green',
                    'medium': 'orange',
                    'low': 'red'
                }.get(confidence, 'black')
                st.caption(f"{confidence_colour}Confidence: {confidence}")

                if result.get('pages_searched'):
                    st.caption(f"Pages Searched: {result['pages_searched']}")

            st.session_state.chat_history.append({'role': 'assistant', 'content': result.get('answer', ''), 'citation': citation, 'confidence': confidence})


        with tab2:
            st.subheader("Document Metadata")
            st.markdown("""
            This section displays metadata about the currently selected document, including its name, description, type, and page count (for PDFs) or line count (for Markdown files).
            """)

            structure = st.session_state.rag.get_structure(st.session_state.current_doc_id)

            st.json(structure)
            
        with tab3:
            st.subheader("Document Information")
            
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Document Name", doc_info.get('doc_name', 'Unknown'))
                st.metric("Document Type", doc_info.get('type', 'Unknown'))
            with col2:
                st.metric("Document ID", st.session_state.current_doc_id[:8] + "...")
                st.metric("Type", doc_info.get('type', 'pdf').upper())
                
            st.write("**Description**")
            st.write(doc_info.get('description', 'No description available.'))

    else:
        st.info("Please upload and select a document to start asking questions.")

        st.subheader("How vectorless RAG works")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **1. Upload Document**: Start by uploading a PDF document. The system will process the document and extract its structure, including page numbers and text content.

            **2. Indexing**: The extracted information is indexed in a way that allows for efficient retrieval without relying on vector embeddings.

            **3. Ask Questions**: Once the document is indexed, you can ask questions about its content. The system will search through the indexed information to find relevant pages and provide answers based on the document's content.

            **4. View Document Info**: You can also view metadata about the document, such as its name, type, description, and page count.
            """)

        with col2:
            st.markdown("""
            **Benefits of Vectorless RAG**:
                        """)
            


if __name__ == "__main__":
    main()