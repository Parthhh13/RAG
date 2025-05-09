from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return a response.
        
        Args:
            query: The user's query string
            
        Returns:
            Dict containing:
                - answer: The final answer
                - context: Any relevant context used
                - metadata: Additional information about the processing
        """
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the type of agent."""
        pass 