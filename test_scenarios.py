"""
Integration Test Suite — Multi-Agent Operations Center
=======================================================
Tests all routing scenarios end-to-end through the LangGraph graph
WITHOUT launching the Streamlit UI.

Run from the project root:
    python test_scenarios.py

Requirements:
  - ADK services running: python adk-services/billing_resolution/agent.py
                          python adk-services/network_diagnostics/agent.py
  - Vector index built (auto-builds on first RAG query)
  - .env with OPENAI_API_KEY set

Expected routing chains:
  1. Billing Dispute      → billing_resolution_adk → customer_comms_crew → FINISH
  2. Network Diagnostics  → network_diagnostics_adk → customer_comms_crew → FINISH
  3. Policy/FAQ           → policy_rag → customer_comms_crew → FINISH
  4. SLA Outage Credit    → network_analytics → policy_rag → customer_comms_crew → FINISH
  5. General Chat/Hi      → general_chat → FINISH
"""

import sys
import os
import uuid
import textwrap

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestration.graph import run_operations_center

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

results = []


def run_scenario(
    scenario_id: int,
    name: str,
    query: str,
    expected_keywords: list[str],
    expect_response: bool = True,
):
    """
    Runs a single scenario against the graph and checks outputs.

    Args:
        scenario_id: Numeric ID shown in output.
        name: Human-readable scenario name.
        query: The user's query string.
        expected_keywords: Strings that must appear in the final response
                           (case-insensitive). Empty list skips keyword check.
        expect_response: If True, the workflow must produce a final response.
    """
    print(f"\n{'='*70}")
    print(f"Scenario {scenario_id}: {name}")
    print(f"Query: {textwrap.shorten(query, 80)}")
    print("─" * 70)

    thread_id = f"test-{scenario_id}-{uuid.uuid4().hex[:8]}"
    passed = True
    notes = []

    try:
        final_state = run_operations_center(query, thread_id=thread_id)
        agent_context = final_state.get("agent_context", "")
        messages = final_state.get("messages", [])

        # Find the final assistant message
        final_response = ""
        for msg in reversed(messages):
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break

        # Check: a response was produced
        has_response = (
            "--- Final Customer Response (CrewAI) ---" in agent_context
            or "--- General Chat Response ---" in agent_context
        )

        if expect_response and not has_response:
            passed = False
            notes.append("No final response marker found in agent_context.")

        # Check: expected keywords present
        for kw in expected_keywords:
            if kw.lower() not in final_response.lower() and kw.lower() not in agent_context.lower():
                notes.append(f"Expected keyword not found: '{kw}'")
                passed = False

        # Print summary
        print("Agent Context Sections:")
        for line in agent_context.split("\n"):
            if line.startswith("---"):
                print(f"  {line}")

        print(f"\nFinal Response Preview:")
        print(textwrap.indent(textwrap.shorten(final_response, 400), "  "))

        status = PASS if passed else FAIL
        print(f"\nResult: {status}")
        if notes:
            for note in notes:
                print(f"  • {note}")

    except Exception as e:
        passed = False
        print(f"Result: {FAIL}")
        print(f"  Exception: {e}")

    results.append({"id": scenario_id, "name": name, "passed": passed})
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  MULTI-AGENT OPERATIONS CENTER — INTEGRATION TEST SUITE")
    print("=" * 70)

    # ── Scenario 1: Billing Dispute (primary test case) ──────────────────────
    run_scenario(
        scenario_id=1,
        name="Billing Dispute — Double Charge + Credit (CUST-10002)",
        query="Customer CUST-10002 was charged twice for Unlimited Plus. Investigate and apply credit.",
        expected_keywords=["CUST-10002", "credit", "duplicate"],
        expect_response=True,
    )

    # ── Scenario 2: Network Diagnostics ──────────────────────────────────────
    run_scenario(
        scenario_id=2,
        name="Network Diagnostics — Tower Status Check",
        query="Check the status of tower TX-512. Is it operational?",
        expected_keywords=["TX-512", "tower"],
        expect_response=True,
    )

    # ── Scenario 3: Policy / FAQ RAG ────────────────────────────────────────
    run_scenario(
        scenario_id=3,
        name="Policy RAG — Trade-in Policy Question",
        query="What is Prodapt's trade-in policy for older devices?",
        expected_keywords=["trade-in", "policy"],
        expect_response=True,
    )

    # ── Scenario 4: SLA Outage Credit (multi-step: analytics → RAG) ─────────
    run_scenario(
        scenario_id=4,
        name="SLA Outage Credit — Multi-Step (Analytics → Policy RAG)",
        query=(
            "We had a 6-hour outage in the Midwest region. "
            "Am I eligible for an SLA credit and what does the policy say?"
        ),
        expected_keywords=["SLA", "outage", "credit"],
        expect_response=True,
    )

    # ── Scenario 5: General Chat — Greeting ─────────────────────────────────
    run_scenario(
        scenario_id=5,
        name="General Chat — Simple Greeting",
        query="Hello! How are you?",
        expected_keywords=[],   # No domain keywords expected for greetings
        expect_response=True,
    )

    # ── Scenario 6: General Chat — History Meta-Question ────────────────────
    run_scenario(
        scenario_id=6,
        name="General Chat — Chat History Meta-Question",
        query="What was the last customer query we discussed?",
        expected_keywords=[],
        expect_response=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    for r in results:
        status = PASS if r["passed"] else FAIL
        print(f"  Scenario {r['id']:>2}: {status}  {r['name']}")

    print(f"\n  Passed: {passed_count}/{total}")

    if passed_count < total:
        print(f"\n  {WARN} Some scenarios failed. Check above for details.")
        sys.exit(1)
    else:
        print(f"\n  {PASS} All scenarios passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
