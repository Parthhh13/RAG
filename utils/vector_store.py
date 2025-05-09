from typing import List, Dict
import os
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class VectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        
    def create_index(self, documents: List[Dict]):
        """Create a FAISS index from documents."""
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
    def save_index(self, directory: str):
        """Save the FAISS index to disk."""
        if self.vector_store is None:
            raise ValueError("No index to save. Create an index first.")
            
        os.makedirs(directory, exist_ok=True)
        self.vector_store.save_local(directory)
        
    def load_index(self, directory: str):
        """Load a FAISS index from disk."""
        self.vector_store = FAISS.load_local(
            directory,
            self.embeddings
        )
        
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        if self.vector_store is None:
            raise ValueError("No index loaded. Load or create an index first.")
            
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in results
        ] 