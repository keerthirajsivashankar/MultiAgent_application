import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
try:
    from crewai.flow.flow import Flow, start, listen, router
except ImportError:
    pass # fallback or will error if flow is strictly required

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

# Ensure the OpenAI API Key is set in the environment
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------
# CrewAI LLM Configuration
# ---------------------------------------------------
# CrewAI version 1.x uses string identifiers or LangChain/LiteLLM configurations.
# Standard "gpt-4o-mini" works directly.
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------
# Agents Definition
# ---------------------------------------------------

communications_specialist = Agent(
    role="Telecom Customer Communications Specialist",
    goal="Draft clear, professional, empathetic, and customer-friendly support responses using the provided operational context.",
    backstory=(
        "You are an expert customer relations specialist at Prodapt. Your job is to take technical data "
        "(like signal strength, outage durations, billing account adjustments, duplicate charge details) "
        "and present it in a beautifully structured, polished, and customer-friendly letter or email. "
        "You avoid technical jargon overload and focus on customer satisfaction."
    ),
    verbose=True,
    allow_delegation=False,
    llm=LLM_MODEL,
)

quality_evaluator_agent = Agent(
    role="Quality Reviewer and Critic",
    goal="Review customer communications for accuracy and tone, or evaluate them to assign a quality score.",
    backstory=(
        "You are a strict QA reviewer and evaluator at Prodapt. "
        "When reviewing drafts, you ensure compliance, professional tone, and 100% accuracy, outputting only the final polished text. "
        "When evaluating responses, you score them based on empathy, clarity, accuracy, and professionalism, outputting ONLY the integer score."
    ),
    verbose=True,
    allow_delegation=False,
    llm=LLM_MODEL,
)

# ---------------------------------------------------
# Core Entry Point Function
# ---------------------------------------------------

def generate_customer_response(user_query: str, agent_context: str) -> str:
    """
    Polishes raw technical findings into a professional, customer-ready response.
    
    Args:
        user_query (str): The original customer question or inquiry.
        agent_context (str): Cumulative outputs and findings from prior specialist agents/databases.
        
    Returns:
        str: Reviewed, professional customer response.
    """
    # Define draft task
    draft_task = Task(
        description=(
            f"Original Customer Inquiry:\n"
            f"'{user_query}'\n\n"
            f"Operational Context / Technical Findings:\n"
            f"{agent_context}\n\n"
            f"Task: Draft a highly professional, polite, and empathetic response to the customer. "
            f"Make sure to explain our findings clearly, tell them what action was taken (or what next steps are), "
            f"and keep the tone reassuring."
        ),
        expected_output="A professional, empathetic draft customer support response explaining the findings and resolution.",
        agent=communications_specialist,
    )

    # Define review task (depends on draft_task)
    review_task = Task(
        description=(
            "Review the drafted customer support response. Ensure that:\n"
            "- The response is factually accurate according to the findings.\n"
            "- The tone is warm, polite, and highly professional.\n"
            "- It does NOT contain any markdown headers or greeting placeholders like [Insert Customer Name].\n"
            "- The final output must contain ONLY the final response ready to be delivered to the customer."
        ),
        expected_output="The final reviewed and refined customer-ready text with no meta-text or placeholders.",
        agent=quality_evaluator_agent,
        context=[draft_task],
    )

    # Assemble the sequential crew
    comms_crew = Crew(
        agents=[communications_specialist, quality_evaluator_agent],
        tasks=[draft_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    # Execute and return the polished string
    result = comms_crew.kickoff()
    return str(result)

def evaluate_customer_response(response_text: str) -> int:
    """
    Evaluates a drafted customer response and returns a score out of 10.
    """
    critic_task = Task(
        description=(
            f"Evaluate the following customer support response:\n\n"
            f"'{response_text}'\n\n"
            f"Score it from 1 to 10 based on empathy, clarity, accuracy, and professionalism. "
            f"Your final answer must be ONLY the integer number. Do not include any other text."
        ),
        expected_output="An integer between 1 and 10 representing the quality score.",
        agent=quality_evaluator_agent,
    )
    
    critic_crew = Crew(
        agents=[quality_evaluator_agent],
        tasks=[critic_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result_str = str(critic_crew.kickoff()).strip()
    try:
        import re
        match = re.search(r'\d+', result_str)
        if match:
            return int(match.group())
        return 0
    except:
        return 0

# ---------------------------------------------------
# CrewAI Flow
# ---------------------------------------------------
class CustomerResponseState(BaseModel):
    user_query: str = ""
    agent_context: str = ""
    response_text: str = ""
    score: int = 0
    iteration: int = 0

class CustomerResponseFlow(Flow[CustomerResponseState]):
    @start()
    def process_drafts(self):
        while True:
            self.state.iteration += 1
            print(f"\n[Flow] Generating draft (Iteration {self.state.iteration})...")
            self.state.response_text = generate_customer_response(self.state.user_query, self.state.agent_context)
            
            print("\n[Flow] Evaluating draft...")
            self.state.score = evaluate_customer_response(self.state.response_text)
            print(f"[Flow] Score: {self.state.score}/10")
            
            if self.state.score >= 8 or self.state.iteration >= 3:
                print("\n[Flow] Draft approved!")
                break
            else:
                print("\n[Flow] Draft rejected, retrying...")
                
        return self.state.response_text

def run_customer_response_flow_direct(user_query: str, agent_context: str) -> str:
    flow = CustomerResponseFlow()
    flow.state.user_query = user_query
    flow.state.agent_context = agent_context
    result = flow.kickoff()
    return str(result)

def run_customer_response_flow(user_query: str, agent_context: str) -> str:
    import json
    import subprocess
    import sys
    import uuid
    
    # 1. Generate unique file paths for isolation
    temp_id = str(uuid.uuid4())
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    input_json_path = os.path.join(data_dir, f"temp_in_{temp_id}.json")
    output_json_path = os.path.join(data_dir, f"temp_out_{temp_id}.json")
    
    try:
        # 2. Write inputs
        input_data = {
            "user_query": user_query,
            "agent_context": agent_context
        }
        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)
            
        # 3. Call python subprocess
        cmd = [
            sys.executable,
            "-m",
            "orchestration.crew_nodes",
            input_json_path,
            output_json_path
        ]
        print(f"\n[Subprocess] Spawning CrewAI flow in separate process...")
        # Force UTF-8 encoding in the subprocess to handle CrewAI's emoji output
        # on Windows where the default codepage (cp1252) cannot encode them.
        import os as _os
        subprocess_env = _os.environ.copy()
        subprocess_env["PYTHONIOENCODING"] = "utf-8"
        subprocess_env["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                cmd, cwd=base_dir, capture_output=True, text=True,
                timeout=300, env=subprocess_env, encoding="utf-8"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "CrewAI subprocess timed out after 300 seconds. "
                "The OpenAI API or CrewAI flow may be hanging. Please retry."
            )
        
        # Log stdout/stderr for trace debugging
        if result.stdout:
            print("[Subprocess Output]:", result.stdout)
        if result.stderr:
            print("[Subprocess Error Output]:", result.stderr)
            
        # 4. Check status and read result
        if result.returncode == 0 and os.path.exists(output_json_path):
            with open(output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            return output_data.get("response_text", "")
        else:
            raise RuntimeError(f"CrewAI subprocess failed with exit code {result.returncode}. Stderr: {result.stderr}")
    finally:
        # 5. Clean up temp files
        for p in [input_json_path, output_json_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

# ---------------------------------------------------
# Self-Test block / Subprocess CLI
# ---------------------------------------------------
if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 2:
        input_json_path = sys.argv[1]
        output_json_path = sys.argv[2]
        try:
            with open(input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            final_output = run_customer_response_flow_direct(data["user_query"], data["agent_context"])
            
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({"response_text": final_output}, f, ensure_ascii=False, indent=2)
            sys.exit(0)
        except Exception as e:
            print(f"Subprocess CLI Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        test_query = "Why is my 5G connection so slow in Austin today?"
        test_context = (
            "[Network Diagnostics ADK]: Status of Tower TX-512 in Austin is OPERATIONAL but experiencing "
            "temporary high traffic load. Performance: signal strength is good (-75 dBm), packet loss is low (0.1%), "
            "but throughput is degraded to 15 Mbps due to active congestion. Recommendation: Reconnect device or "
            "wait for NOC peak traffic load to settle by 8 PM."
        )
        print("\n--- Running CrewAI Self-Test ---")
        final_output = run_customer_response_flow_direct(test_query, test_context)
        print("\n--- Final Customer-Ready Response ---")
        print(final_output)
