from orchestration.adk_remote_client import (
    run_network_diagnostics,
    run_billing_resolution,
)

print("\n===== NETWORK TEST =====\n")

result1 = run_network_diagnostics(
    "Check tower TX-512 status"
)

print("NETWORK RESULT:")
print(result1)

print("\n===== BILLING TEST =====\n")

result2 = run_billing_resolution(
    "CUST-10002 was charged twice"
)

print("BILLING RESULT:")
print(result2)