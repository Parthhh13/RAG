from typing import Dict, Any, List
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base_agent import BaseAgent
from utils.vector_store import VectorStore

class RAGAgent(BaseAgent):
    def __init__(self, vector_store: VectorStore, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.vector_store = vector_store
        
        # Initialize the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Create the pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create the LLM
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the user's question.
            If you cannot find the answer in the context, say so. Do not make up information.
            
            Context:
            {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])
        
    def process_query(self, query: str) -> Dict[str, Any]:
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate response using LLM
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        # Handle both string and object responses
        answer = response if isinstance(response, str) else response.content
        
        return {
            "answer": answer,
            "context": relevant_docs,
            "metadata": {
                "agent_type": self.get_agent_type(),
                "num_docs_retrieved": len(relevant_docs)
            }
        }
    
    def get_agent_type(self) -> str:
        return "RAG" 