from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State definition for the LangGraph supervisor orchestration.
    """
    # Conversation history: append-only across nodes
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # The supervisor's routing decision (next node to invoke or FINISH)
    next: str
    
    # The original customer query/question
    user_query: str
    
    # Accumulated text outputs from all worker nodes, eventually passed to CrewAI
    agent_context: str
    
    # Counter to prevent infinite loops in graph
    loop_count: int
    
    # Counter for how many times the customer response has been generated
    generation_count: int
    
    # Score out of 10 given by the Critic Agent
    critic_score: int
