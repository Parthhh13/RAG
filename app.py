import os
import streamlit as st
from utils.document_loader import DocumentLoader
from utils.vector_store import VectorStore
from agents.rag_agent import RAGAgent

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None

def initialize_system():
    """Initialize the RAG system with documents."""
    if st.session_state.vector_store is None:
        # Create vector store
        vector_store = VectorStore()
        
        # Load and process documents
        loader = DocumentLoader()
        documents = loader.load_directory("data")
        
        if documents:
            vector_store.create_index(documents)
            st.session_state.vector_store = vector_store
            st.session_state.rag_agent = RAGAgent(vector_store)
            return True
    return False

def main():
    st.title("RAG-Powered Q&A Assistant")
    
    # Initialize system if needed
    if not st.session_state.vector_store:
        with st.spinner("Initializing system..."):
            if not initialize_system():
                st.error("No documents found in the data directory. Please add some documents first.")
                return
    
    # Query input
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Processing your question..."):
            # Process query using RAG agent
            response = st.session_state.rag_agent.process_query(query)
            
            # Display results
            st.subheader("Answer")
            st.write(response["answer"])
            
            # Display context
            st.subheader("Retrieved Context")
            for doc in response["context"]:
                with st.expander(f"Source: {doc['metadata']['source']} (Score: {doc['score']:.2f})"):
                    st.write(doc['content'])
            
            # Display metadata
            st.subheader("Processing Information")
            st.json(response["metadata"])

if __name__ == "__main__":
    main() 