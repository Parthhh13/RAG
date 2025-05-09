# RAG-Powered Multi-Agent Q&A Assistant

This project implements a knowledge assistant that combines Retrieval-Augmented Generation (RAG) with an agentic workflow to provide intelligent answers to user queries.

## Features

- Document ingestion and chunking
- Vector-based semantic search using FAISS
- LLM-powered answer generation
- Agentic workflow for specialized queries
- Interactive web interface using Streamlit

## Architecture

The system consists of several key components:

1. **Data Ingestion Module**: Processes and chunks documents for indexing
2. **Vector Store**: FAISS-based vector database for semantic search
3. **LLM Integration**: OpenAI integration for answer generation
4. **Agent System**: Routes queries to appropriate tools or RAG pipeline
5. **Web Interface**: Streamlit-based UI for interaction

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Place your documents in the `data` directory
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
.
├── app.py                 # Streamlit web interface
├── agents/               # Agent system implementation
│   ├── __init__.py
│   ├── base_agent.py
│   └── rag_agent.py
├── data/                 # Document storage
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── document_loader.py
│   └── vector_store.py
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the application using `streamlit run app.py`
2. Enter your question in the text input
3. View the agent's decision process and final answer
4. See the retrieved context snippets used to generate the answer

## Design Choices

- **FAISS**: Chosen for its efficient similarity search capabilities
- **Streamlit**: Selected for rapid UI development and easy deployment
- **LangChain**: Used for agent orchestration and LLM integration
- **Chunking Strategy**: Documents are split into 500-token chunks with overlap 