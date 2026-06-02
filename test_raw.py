"""
Quick smoke-test for the ADK remote client wrappers.
Run from the project root:
    python test_raw.py
Both ADK services must be running (ports 8001 and 8002).
"""
from orchestration.adk_remote_client import (
    run_network_diagnostics,
    run_billing_resolution,
)

print("===== NETWORK DIAGNOSTICS TEST =====")
result = run_network_diagnostics("Check tower TX-512 status")
print("RESULT:", result)

print("\n===== BILLING RESOLUTION TEST =====")
result = run_billing_resolution(
    "Customer CUST-10002 was charged twice for Unlimited Plus. Investigate and apply credit."
)
print("RESULT:", result)