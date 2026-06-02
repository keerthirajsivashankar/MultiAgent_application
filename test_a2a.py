import requests

print("NETWORK CARD:")
print(requests.get("http://localhost:8001/.well-known/agent-card.json").text)

print("\nBILLING CARD:")
print(requests.get("http://localhost:8002/.well-known/agent-card.json").text)