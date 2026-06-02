import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

# Import the shared state
from orchestration.state import AgentState

# Import the worker functions
from llamaindex_rag.document_rag import ask_question as run_policy_rag
from llamaindex_rag.sql_semantic_search import ask_sql_question as run_network_analytics
from orchestration.adk_remote_client import run_network_diagnostics, run_billing_resolution
from orchestration.crew_nodes import run_customer_response_flow

# Load environment variables
load_dotenv()

# Set up standard ChatOpenAI LLM for the supervisor
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================
# Supervisor Node Definition
# =====================================================

# Define structured output schema for routing decision
class RouteResponse(BaseModel):
    next: Literal[
        "policy_rag",
        "network_analytics",
        "network_diagnostics_adk",
        "billing_resolution_adk",
        "customer_comms_crew",
        "general_chat",
        "FINISH"
    ] = Field(
        description="Pick the next specialist worker. For greetings, chit-chat, or questions about previous messages in the chat history, select 'general_chat'. Once all data is gathered, select 'customer_comms_crew'. Once the response is drafted, select 'FINISH'."
    )
    rationale: str = Field(
        description="Brief reasoning behind selecting the next specialist."
    )

def supervisor_node(state: AgentState):
    """
    Supervisor node that analyzes state, history, and accumulated context
    to select the next worker or choose FINISH.
    """
    current_loops = state.get("loop_count", 0)
    agent_context = state.get("agent_context", "")

    # Fail-safe: if the customer response is already drafted, force FINISH to prevent looping
    has_final_response = (
        "--- Final Customer Response (CrewAI) ---" in agent_context
        or "--- General Chat Response ---" in agent_context
    )
    if has_final_response:
        print("\n[Supervisor] Final response already drafted. Bypassing LLM and routing to FINISH.")
        return {"next": "FINISH", "loop_count": current_loops + 1}

    if current_loops >= 10:
        print("\n[Supervisor] Loop limit reached. Forcing escalation to FINISH.")
        return {
            "next": "FINISH",
            "loop_count": current_loops + 1,
            "agent_context": agent_context + "\n\n--- ESCALATION ---\nMax loop limit (10) reached. Forcing response generation."
        }

    # System instruction for supervisor behavior
    system_prompt = (
        "You are the Prodapt AI Operations Supervisor. Your job is to analyze the customer's query, "
        "historical messages, and the current accumulated operational context to decide which specialist "
        "worker node to call next.\n\n"
        "Available specialists and their responsibilities:\n"
        "1. 'policy_rag': Query billing dispute procedures, trade-in policy, European roaming zone prices, MTTR, etc.\n"
        "2. 'network_analytics': Perform analytics across SQLite tables (e.g. counting failures in regions, finding towers with highest packet loss).\n"
        "3. 'network_diagnostics_adk': Diagnose active network / tower connectivity issues (requires tower_id).\n"
        "4. 'billing_resolution_adk': Investigate billing disputes (lookup accounts, find duplicate charges, apply credits).\n"
        "5. 'customer_comms_crew': Always call this worker LAST to polish technical outputs from specialists into an empathetic, customer-ready letter.\n"
        "6. 'general_chat': Call this for greetings, general conversational questions (e.g. 'hi'), or questions asking about the chat history (e.g. 'what was the last question?', 'what customer id did we discuss?').\n\n"
        "Rules:\n"
        "- If the request involves multiple aspects, chain them sequentially.\n"
        "- If the query is just a greeting, conversational chit-chat, or asking about the history of the current chat, route immediately to 'general_chat'.\n"
        "- Once you have sufficient technical information to answer the customer, you MUST route to 'customer_comms_crew' to draft the response.\n"
        "- SLA Credit Outage Inquiries: When a user asks about SLA credit eligibility or rules due to an outage (e.g. 'We had a 6-hour outage in the Midwest. Am I eligible for an SLA credit and what does policy say?'), you MUST follow this sequential path: first, route to 'network_analytics' to query the SQL database tables for the verified outage details (like duration and region); second, once you have the outage details in findings, route to 'policy_rag' to search the SLA policy documents for eligibility; third, route to 'customer_comms_crew' to draft the response. Do NOT route directly to 'billing_resolution_adk' for this verification process.\n"
        "- CRITICAL: If you see '--- Final Customer Response (CrewAI) ---' or '--- General Chat Response ---' in the Accumulated Findings, you MUST route to 'FINISH' to end the workflow. Do NOT call 'customer_comms_crew' or 'general_chat' twice."
    )

    latest_query = state.get("user_query", "")
    
    # Contextualize query if there is history
    history = state.get("messages", [])[:-1]
    if len(history) > 0 and latest_query:
        contextualize_system_prompt = (
            "Given the following chat history and a follow-up question, "
            "reformulate the follow-up question into a standalone question that "
            "captures the full context of the conversation. "
            "Important: Do NOT change the casing of severity levels (e.g. keep 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW' in uppercase), "
            "technology types (e.g. '5G', 'LTE'), or specific IDs.\n"
            "Return ONLY the reformulated question text. Do not add any greeting or meta-text."
        )
        contextualize_messages = [
            {"role": "system", "content": contextualize_system_prompt}
        ]
        for msg in history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            contextualize_messages.append({"role": role, "content": msg.content})
        
        contextualize_messages.append({"role": "user", "content": latest_query})
        
        response = llm.invoke(contextualize_messages)
        contextualized_query = str(response.content).strip()
        print(f"\n[Supervisor Contextualization] Original: '{latest_query}' -> Contextualized: '{contextualized_query}'")
        latest_query = contextualized_query

    # Format the message history
    formatted_messages = []
    for idx, msg in enumerate(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            # Use contextualized query for the last user message if applicable
            content = latest_query if idx == len(state.get("messages", [])) - 1 else msg.content
            formatted_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
            
    # Inject system prompt with the current turn's accumulated findings
    findings_prompt = f"\n\nAccumulated Findings for the current query:\n{agent_context}"
    formatted_messages.insert(0, {"role": "system", "content": system_prompt + findings_prompt})

    # Call LLM with structured output
    structured_llm = llm.with_structured_output(RouteResponse)
    decision = structured_llm.invoke(formatted_messages)

    # Guardrail: Prevent premature FINISH before a response has been drafted.
    # Only override FINISH→customer_comms_crew when:
    #   (a) no final response has been drafted yet, AND
    #   (b) there IS meaningful context for CrewAI to work with.
    # If context is empty (e.g. ADK service unreachable) let FINISH through
    # so the graph doesn't loop endlessly with nothing to send to CrewAI.
    has_final_response = (
        "--- Final Customer Response (CrewAI) ---" in agent_context
        or "--- General Chat Response ---" in agent_context
    )
    has_context = bool(agent_context.strip())
    if decision.next == "FINISH" and not has_final_response and has_context:
        print("\n[Supervisor Guardrail] Overriding premature FINISH → customer_comms_crew (context exists, no response yet).")
        decision.next = "customer_comms_crew"
    elif decision.next == "FINISH" and not has_final_response and not has_context:
        print("\n[Supervisor Guardrail] FINISH with no context — allowing through to avoid empty CrewAI loop.")

    print(f"\n[Supervisor Decision] Next: {decision.next} | Rationale: {decision.rationale}")

    return {"next": decision.next, "loop_count": current_loops + 1, "user_query": latest_query}

# =====================================================
# Specialist Worker Nodes
# =====================================================

def policy_rag_node(state: AgentState):
    """Worker node calling LlamaIndex RAG for policy details."""
    query = state["user_query"]
    print(f"\n[Worker] Calling Policy RAG for: {query}")
    
    # Call policy RAG
    result = run_policy_rag(query)
    
    # Update context and history
    context_update = f"\n--- Policy Findings ---\n{result}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=f"[Policy RAG]: {result}")]
    }

def network_analytics_node(state: AgentState):
    """Worker node calling LlamaIndex Semantic SQL query engine."""
    query = state["user_query"]
    print(f"\n[Worker] Calling Network Analytics for: {query}")
    
    # Call semantic SQL
    result = run_network_analytics(query)
    
    # Update context and history
    context_update = f"\n--- Network Analytics SQL Findings ---\n{result}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=f"[Network Analytics]: {result}")]
    }

def network_diagnostics_adk_node(state: AgentState):
    """Worker node calling the remote ADK Network Diagnostics service."""
    query = state["user_query"]
    print(f"\n[Worker] Calling Network Diagnostics ADK for: {query}")
    
    # Call remote ADK
    result = run_network_diagnostics(query)
    
    # Update context and history
    context_update = f"\n--- Live Network Diagnostics (ADK) Findings ---\n{result}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=f"[Network Diagnostics ADK]: {result}")]
    }

def billing_resolution_adk_node(state: AgentState):
    """Worker node calling the remote ADK Billing Resolution service."""
    query = state["user_query"]
    print(f"\n[Worker] Calling Billing Resolution ADK for: {query}")
    
    # Call remote ADK
    result = run_billing_resolution(query)
    
    # Update context and history
    context_update = f"\n--- Billing Resolution (ADK) Findings ---\n{result}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=f"[Billing Resolution ADK]: {result}")]
    }

def customer_comms_crew_node(state: AgentState, config: RunnableConfig):
    """Worker node calling the CrewAI sequentially-polishing crew Flow."""
    print("\n[Worker] Calling Customer Communications CrewAI Flow...")
    
    # Call CrewAI flow to draft, review, and evaluate sequentially
    result = run_customer_response_flow(state["user_query"], state["agent_context"])
    
    # Extract thread ID
    thread_id = "default_thread"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", "default_thread")
        
    # Log user query and assistant response to past_conversations SQL table
    try:
        conn = sqlite3.connect("data/telecom_ops.db")
        cursor = conn.cursor()
        
        # Log User Message
        cursor.execute(
            "INSERT INTO past_conversations (thread_id, role, content) VALUES (?, 'user', ?)",
            (thread_id, state["user_query"])
        )
        
        # Log Assistant Response
        cursor.execute(
            "INSERT INTO past_conversations (thread_id, role, content) VALUES (?, 'assistant', ?)",
            (thread_id, result)
        )
        
        conn.commit()
        conn.close()
        print(f"[customer_comms_crew_node] Successfully logged turn to past_conversations table for thread: {thread_id}")
    except Exception as e:
        print(f"[customer_comms_crew_node] Failed to log past conversations: {e}")
        
    # Update context and history
    context_update = f"\n--- Final Customer Response (CrewAI) ---\n{result}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=result)]
    }

def general_chat_node(state: AgentState, config: RunnableConfig):
    """Worker node to handle greetings, general chit-chat, and chat history meta-questions."""
    print("\n[Worker] Calling General Chat Node...")
    
    system_prompt = (
        "You are the Prodapt AI Operations Assistant. Answer the user's query. "
        "If they are making general conversation (greetings, small talk) or asking about previous messages "
        "if any other queries asked please deny those question respectfully"
        "you should not answer any other queries rather than the queries regarding the company , in this chat, respond politely, helpfully, and accurately using the conversation history."
    )
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
            
    response = llm.invoke(formatted_messages)
    ans = str(response.content).strip()
    
    # Extract thread ID
    thread_id = "default_thread"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", "default_thread")
        
    # Log user query and assistant response to past_conversations SQL table
    try:
        conn = sqlite3.connect("data/telecom_ops.db")
        cursor = conn.cursor()
        
        # Log User Message
        cursor.execute(
            "INSERT INTO past_conversations (thread_id, role, content) VALUES (?, 'user', ?)",
            (thread_id, state["user_query"])
        )
        
        # Log Assistant Response
        cursor.execute(
            "INSERT INTO past_conversations (thread_id, role, content) VALUES (?, 'assistant', ?)",
            (thread_id, ans)
        )
        
        conn.commit()
        conn.close()
        print(f"[general_chat_node] Successfully logged turn to past_conversations table for thread: {thread_id}")
    except Exception as e:
        print(f"[general_chat_node] Failed to log past conversations: {e}")
        
    context_update = f"\n--- General Chat Response ---\n{ans}\n"
    return {
        "agent_context": state["agent_context"] + context_update,
        "messages": [AIMessage(content=ans)]
    }

# =====================================================
# Graph Construction & Compilation
# =====================================================

# Initialize the workflow
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("policy_rag", policy_rag_node)
workflow.add_node("network_analytics", network_analytics_node)
workflow.add_node("network_diagnostics_adk", network_diagnostics_adk_node)
workflow.add_node("billing_resolution_adk", billing_resolution_adk_node)
workflow.add_node("customer_comms_crew", customer_comms_crew_node)
workflow.add_node("general_chat", general_chat_node)

# Add edges back to supervisor
workflow.add_edge("policy_rag", "supervisor")
workflow.add_edge("network_analytics", "supervisor")
workflow.add_edge("network_diagnostics_adk", "supervisor")
workflow.add_edge("billing_resolution_adk", "supervisor")
workflow.add_edge("customer_comms_crew", "supervisor")
workflow.add_edge("general_chat", "supervisor")

# Configure supervisor routing matrix
def route_decision(state: AgentState):
    """Determines which node the supervisor chooses next."""
    destination = state["next"]
    if destination == "FINISH":
        return END
    return destination

# Add the conditional supervisor edges
workflow.add_conditional_edges(
    "supervisor",
    route_decision
)

# Set the entry point
workflow.set_entry_point("supervisor")

# Compile the graph with SqliteSaver checkpointer for persistent stateful conversation threads
checkpoint_conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
memory = SqliteSaver(checkpoint_conn)
compiled_graph = workflow.compile(checkpointer=memory)

# =====================================================
# Main execution helper for UI / testing
# =====================================================
def run_operations_center(query: str, thread_id: str = "default_thread") -> dict:
    """
    Synchronously executes the compiled multi-agent graph with user query.
    
    Args:
        query (str): The raw inquiry from the customer/user.
        thread_id (str): Unique identifier for the conversation session.
        
    Returns:
        dict: Final state output from the graph.
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "agent_context": "",
        "next": "supervisor",
        "loop_count": 0,
        "generation_count": 0,
        "critic_score": 0
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    final_state = compiled_graph.invoke(initial_state, config=config)
    return final_state
