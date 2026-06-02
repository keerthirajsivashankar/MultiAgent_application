# How to Run the Prodapt AI Operations Center

Follow these instructions to run the project successfully locally. The project relies on a SQLite database, two background Google ADK agent servers, and a Streamlit UI.

## 1. Prerequisites and Environment Setup

Make sure you have Python installed. The project already contains a `venv` directory.
Activate the virtual environment:
- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

> **⚠️ IMPORTANT — Always use the venv Python.**
> All project dependencies (`llama-index`, `google-adk`, `crewai`, etc.) are installed inside the `venv`.
> Running any project script with the **system `python`** will fail with `ModuleNotFoundError`.
> Always either:
> - Activate the venv first (`.\venv\Scripts\activate`) and then use `python`, OR
> - Use the explicit venv interpreter: `venv\Scripts\python.exe <script>` (Windows) / `venv/bin/python <script>` (Mac/Linux)

*(If you need to reinstall dependencies for any reason, run `pip install -r requirements.txt` after activating).*

Ensure your `.env` file is properly configured with your OpenAI API key in the root of the project:
```env
OPENAI_API_KEY=your-api-key-here
```

## 2. Initialize the Database
The project uses a local SQLite database located in `data/telecom_ops.db`. If the database is missing or you need to recreate it from scratch, run the initialization script:
```bash
python create_db.py
```

## 3. Start the Google ADK Agent Services
You must run the two ADK backend services before you start the UI, otherwise the orchestration graph will fail to communicate with them (throwing a 503 error).

Open **two new terminal windows**, activate the `venv` in each, and start the servers:

**Terminal 1 (Network Diagnostics - Port 8001):**
```bash
python adk-services/network_diagnostics/agent.py
```

**Terminal 2 (Billing Resolution - Port 8002):**
```bash
python adk-services/billing_resolution/agent.py
```

Leave both of these terminal windows open and running.

## 4. Run the Streamlit User Interface
With the database ready and both ADK servers running on ports `8001` and `8002`, you can now start the main LangGraph operations center UI. 

Open a **third terminal**, ensure the `venv` is active, and run:
```bash
streamlit run ui/app.py
```

This will automatically open the web application in your browser (typically at `http://localhost:8501`). The sidebar will indicate if your database and both ADK services are successfully connected. You can then submit your inquiries directly into the interface!

## 5. Run Integration Tests (without UI)

To verify all routing scenarios work end-to-end without launching Streamlit, make sure the two ADK services are running (Step 3), then run:

```bash
python test_scenarios.py
```

This will exercise all 6 scenarios:
| # | Scenario | Expected Route |
|---|---|---|
| 1 | Billing dispute (CUST-10002 double charge) | `billing_resolution_adk` → `customer_comms_crew` |
| 2 | Network tower status check | `network_diagnostics_adk` → `customer_comms_crew` |
| 3 | Policy/FAQ (trade-in policy) | `policy_rag` → `customer_comms_crew` |
| 4 | SLA outage credit (multi-step) | `network_analytics` → `policy_rag` → `customer_comms_crew` |
| 5 | General greeting | `general_chat` |
| 6 | Chat history meta-question | `general_chat` |

To run a quick ADK connectivity smoke test:
```bash
python test_raw.py          # tests ADK remote client wrappers
python test_remote_clinet.py  # tests full ADK A2A round-trip
```
