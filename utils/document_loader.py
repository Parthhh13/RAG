from typing import List, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Load and chunk a document based on its file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Select appropriate loader based on file type
        if file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load and split the document
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Convert chunks to dictionary format
        return [
            {
                'content': chunk.page_content,
                'metadata': {
                    'source': chunk.metadata.get('source', file_path),
                    'page': chunk.metadata.get('page', 0)
                }
            }
            for chunk in chunks
        ]
    
    def load_directory(self, directory_path: str) -> List[Dict]:
        """Load and chunk all supported documents in a directory."""
        all_chunks = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    chunks = self.load_document(file_path)
                    all_chunks.extend(chunks)
                except ValueError as e:
                    print(f"Skipping {filename}: {str(e)}")
                    
        return all_chunks 