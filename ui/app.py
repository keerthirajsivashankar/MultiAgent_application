import streamlit as st
import os
import requests
import sys
import uuid

# Ensure the root project directory is in the path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from orchestration.graph import compiled_graph

st.set_page_config(page_title="Prodapt AI Operations Center", page_icon="📶", layout="wide")

# ==========================================
# Sidebar: System Status & Framework Map
# ==========================================
with st.sidebar:
    st.header("System Status")
    
    # Check Database
    db_path = os.path.join("data", "telecom_ops.db")
    if os.path.exists(db_path):
        st.success(f"Database: Ready")
    else:
        st.error("Database: Missing. Please run init scripts.")

    # Check Vector Index
    index_path = os.path.join("data", "vector_index")
    if os.path.exists(index_path):
        st.success(f"Vector Index: Ready")
    else:
        st.info("Vector Index: Missing (Will build on first RAG query)")

    # Check ADK Services
    def check_adk_service(port):
        try:
            res = requests.get(f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=1)
            return res.status_code == 200
        except Exception:
            return False

    adk_8001_ready = check_adk_service(8001)
    if adk_8001_ready:
        st.success("Network Diagnostics (8001): Running")
    else:
        st.error("Network Diagnostics (8001): Not running")
        
    adk_8002_ready = check_adk_service(8002)
    if adk_8002_ready:
        st.success("Billing Resolution (8002): Running")
    else:
        st.error("Billing Resolution (8002): Not running")
        
    if not (adk_8001_ready and adk_8002_ready):
        st.warning("⚠️ Start both ADK services via `python adk-services/<agent_dir>/agent.py` before querying.")

    st.markdown("---")
    st.header("Framework Map")
    st.markdown("""
    | Capability | Framework |
    |---|---|
    | Policy/FAQ | LlamaIndex RAG |
    | Analytics | LlamaIndex SQL |
    | Network | Google ADK (A2A) |
    | Billing | Google ADK (A2A) |
    | Comms | CrewAI |
    | Orchestration | LangGraph |
    """)

    st.markdown("---")
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.session_state.last_trace = []
        st.rerun()

# ==========================================
# Main Layout
# ==========================================
st.title("Prodapt AI Operations Center")
st.caption("Powered by LangGraph, LlamaIndex, Google ADK, and CrewAI")

# Initialize Session States
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

# Display Chat History
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            st.markdown(
                f"<div style='font-size: 0.85em; color: #a1a1aa; border-top: 1px solid #3f3f46; margin-top: 10px; padding-top: 8px; font-family: monospace;'>"
                f"🛡️ <b>Execution Path:</b> {msg['trace']}"
                f"</div>",
                unsafe_allow_html=True
            )

# Display last execution trace if available
if st.session_state.last_trace:
    with st.expander("📋 Last Agent Execution Trace", expanded=False):
        for step in st.session_state.last_trace:
            st.markdown(f"**Step {step['step']}: `{step['worker']}`**")
            display_text = step["output"]
            if len(display_text) > 400:
                display_text = display_text[:400] + " ... [TRUNCATED]"
            st.code(display_text, language="markdown")

# Chat input from user
if prompt := st.chat_input("How can I assist you with Prodapt Operations today?"):
    # 1. Display user message in chat immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    # 2. Setup traces UI
    trace_container = st.container()
    with trace_container:
        trace_expander = st.expander("🛠️ Live Agent Execution Trace", expanded=True)
    
    trace_data = []
    path_nodes = []
    final_response_text = None
    
    with st.spinner("Analyzing and routing..."):
        try:
            # Setup LangGraph initial state updates
            # Since checkpointer is active, this will append to the existing thread state.
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "user_query": prompt,
                "agent_context": "",
                "next": "supervisor",
                "loop_count": 0,
                "generation_count": 0,
                "critic_score": 0
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            step_num = 1
            for output in compiled_graph.stream(initial_state, config=config):
                for node_name, state_update in output.items():
                    if node_name == "supervisor":
                        next_node = state_update.get("next", "")
                        if next_node == "FINISH":
                            path_nodes.append("FINISH")
                        else:
                            if not path_nodes or path_nodes[-1] != "Supervisor":
                                path_nodes.append("Supervisor")
                        continue
                        
                    worker_output = ""
                    if "messages" in state_update and len(state_update["messages"]) > 0:
                        worker_output = state_update["messages"][-1].content
                    
                    trace_data.append({
                        "step": step_num,
                        "worker": node_name,
                        "output": worker_output
                    })
                    
                    # Map specialist worker nodes to display names
                    display_names = {
                        "network_analytics": "NetworkAnalytics",
                        "policy_rag": "PolicyRAG",
                        "network_diagnostics_adk": "NetworkDiagnostics",
                        "billing_resolution_adk": "BillingResolution",
                        "customer_comms_crew": "CustomerCommsCrew",
                        "general_chat": "GeneralChat",
                    }
                    display_name = display_names.get(node_name, node_name)
                    if not path_nodes or path_nodes[-1] != display_name:
                        path_nodes.append(display_name)
                    
                    # Update live trace in UI
                    with trace_expander:
                        st.markdown(f"**Step {step_num}: `{node_name}`**")
                        display_text = worker_output
                        if len(display_text) > 400:
                            display_text = display_text[:400] + " ... [TRUNCATED]"
                        st.code(display_text, language="markdown")
                        
                    if node_name in ("customer_comms_crew", "general_chat"):
                        final_response_text = worker_output
                        
                    step_num += 1
                    
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            
    # 3. Store trace & final response
    st.session_state.last_trace = trace_data
    
    if final_response_text:
        trace_str = " → ".join(path_nodes)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": final_response_text,
            "trace": trace_str
        })
        st.rerun()
    else:
        if len(trace_data) > 0:
            st.warning("The workflow completed, but no final customer response was drafted.")
        else:
            st.info("No workers ran. Could not route the query.")
