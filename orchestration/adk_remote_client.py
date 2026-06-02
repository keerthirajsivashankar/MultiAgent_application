import asyncio
import uuid
import httpx
import concurrent.futures

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.genai.types import Content, Part

# =====================================================
# Agent Card URLs
# =====================================================

NETWORK_AGENT_CARD = "http://127.0.0.1:8001/.well-known/agent-card.json"
BILLING_AGENT_CARD = "http://127.0.0.1:8002/.well-known/agent-card.json"

# =====================================================
# Core Async Executor
# =====================================================

async def run_remote_a2a(agent_name: str, agent_card: str, message: str) -> str:
    """
    Generic executor for ANY ADK RemoteA2aAgent.
    Performs a fast pre-flight check before connecting to avoid ADK's
    internal retry loop when the service is not running.
    """
    print("CALLING:", agent_card)

    # ── Pre-flight: check if the agent card endpoint is reachable ────────────
    # RemoteA2aAgent retries the agent card fetch ~6 times internally.
    # A single fast httpx check here prevents that long wait.
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            preflight = await client.get(agent_card)
            if preflight.status_code != 200:
                return (
                    f"ADK service at {agent_card} returned HTTP {preflight.status_code}. "
                    "Ensure the ADK service is running on the correct port."
                )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, OSError):
        return (
            "ADK service is not reachable. "
            "Ensure ports 8001 and 8002 are running before submitting queries."
        )

    # ── Main A2A execution ────────────────────────────────────────────────────
    try:
        agent = RemoteA2aAgent(
            name=agent_name,
            agent_card=agent_card,
        )
        session_service = InMemorySessionService()

        runner = Runner(
            agent=agent,
            app_name="capstone_project",
            session_service=session_service,
        )

        session = await session_service.create_session(
            app_name="capstone_project",
            user_id=str(uuid.uuid4()),
        )

        events = runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=Content(
                role="user",
                parts=[Part(text=message)],
            ),
        )

        final_text = None

        async for event in events:
            if event.error_message:
                return f"Remote ADK execution error: {event.error_message}"
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text

        return final_text or "No response received from remote agent."

    except httpx.ConnectError:
        return (
            "ADK service is not reachable. "
            "Ensure ports 8001 and 8002 are running."
        )

    except httpx.ConnectTimeout:
        return (
            "ADK service timeout. The agent took too long to respond."
        )

    except Exception as e:
        return f"Remote ADK execution error: {str(e)}"


# =====================================================
# Safe Sync Runner — works inside Streamlit AND plain Python
# =====================================================

def _run_in_new_loop(coro) -> str:
    """
    Runs an async coroutine in a brand-new event loop inside a
    background thread. This avoids the "event loop already running"
    RuntimeError that occurs when asyncio.run() is called from within
    Streamlit (which owns its own event loop).

    Works safely from:
      - Streamlit (async event loop already running)
      - Plain Python scripts (no event loop)
      - LangGraph graph nodes
    """
    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Shutdown pending async generators to suppress
            # OpenTelemetry ContextVar GeneratorExit warnings
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_thread_target)
        return future.result()



# =====================================================
# Sync Wrappers (LangGraph-friendly)
# =====================================================

def run_network_diagnostics(question: str) -> str:
    """
    LangGraph node wrapper — Network Diagnostics ADK.
    Safe to call from Streamlit or plain Python.
    """
    return _run_in_new_loop(
        run_remote_a2a("network_diagnostics_remote", NETWORK_AGENT_CARD, question)
    )


def run_billing_resolution(question: str) -> str:
    """
    LangGraph node wrapper — Billing ADK.
    Safe to call from Streamlit or plain Python.
    """
    return _run_in_new_loop(
        run_remote_a2a("billing_resolution_remote", BILLING_AGENT_CARD, question)
    )


# =====================================================
# Structured Wrappers (for trace/debug UI)
# =====================================================

def run_network_diagnostics_result(question: str) -> dict:
    """
    Returns structured output for LangGraph trace UI.
    """
    result = run_network_diagnostics(question)
    return {
        "success": "ADK service" not in result,
        "response": result,
    }


def run_billing_resolution_result(question: str) -> dict:
    """
    Returns structured output for LangGraph trace UI.
    """
    result = run_billing_resolution(question)
    return {
        "success": "ADK service" not in result,
        "response": result,
    }