import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

from llama_index.core import SQLDatabase
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.llms.openai import OpenAI

# ----------------------------------------
# Load Environment Variables
# ----------------------------------------
load_dotenv()

# ----------------------------------------
# Paths
# ----------------------------------------
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../data/telecom_ops.db"
    )
)

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

print("DATABASE_PATH =", DATABASE_PATH)
print("EXISTS =", os.path.exists(DATABASE_PATH))
# ----------------------------------------
# Embedding Model
# ----------------------------------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------------
# OpenAI LLM
# ----------------------------------------
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

Settings.embed_model = embed_model
Settings.llm = llm

# ----------------------------------------
# SQLAlchemy Engine
# ----------------------------------------
engine = create_engine(DATABASE_URL)

# ----------------------------------------
# LlamaIndex SQL Database
# ----------------------------------------
sql_database = SQLDatabase(engine)

# ----------------------------------------
# SQL Table Node Mapping
# ----------------------------------------
table_node_mapping = SQLTableNodeMapping(sql_database)

# ----------------------------------------
# SQL Table Schemas
# Context strings help semantic retrieval
# ----------------------------------------
table_schema_objs = [

    SQLTableSchema(
        table_name="network_towers",
        context_str=(
            "Contains telecom tower information including "
            "tower region, city, technology type (values: 'LTE', '5G' in uppercase), operational "
            "status (values: 'OPERATIONAL', 'DEGRADED', 'OUTAGE', 'MAINTENANCE' in uppercase), maintenance dates, and subscriber capacity."
        ),
    ),

    SQLTableSchema(
        table_name="network_outages",
        context_str=(
            "Contains historical telecom network outages, "
            "severity levels (values: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW' in uppercase), outage duration, affected customers, "
            "root causes, and related incident IDs."
        ),
    ),

    SQLTableSchema(
        table_name="tower_performance",
        context_str=(
            "Contains telecom tower performance metrics including "
            "latency, packet loss, throughput, signal strength, "
            "and performance timestamps."
        ),
    ),

    SQLTableSchema(
        table_name="open_incidents",
        context_str=(
            "Contains active telecom network incidents including "
            "incident severity (values: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW' in uppercase), resolution status (values: 'OPEN', 'IN_PROGRESS', 'RESOLVED' in uppercase), ETA, and "
            "incident descriptions."
        ),
    ),

    SQLTableSchema(
        table_name="customer_subscriptions",
        context_str=(
            "Contains customer subscription plans, account types (values: 'Consumer', 'Business', 'Enterprise'), "
            "monthly fees, data limits, contract end dates, "
            "and customer regions."
        ),
    ),

    SQLTableSchema(
        table_name="billing_accounts",
        context_str=(
            "Contains customer billing account balances and "
            "billing cycle information."
        ),
    ),

    SQLTableSchema(
        table_name="billing_charges",
        context_str=(
            "Contains customer billing charges, charge descriptions, "
            "billing periods, duplicate charge flags, and charge dates."
        ),
    ),

    SQLTableSchema(
        table_name="billing_credits",
        context_str=(
            "Contains billing credits, refunds, adjustment reasons, "
            "credit approval status (values: 'APPLIED', 'PENDING_APPROVAL' in uppercase), and applied timestamps."
        ),
    ),

    SQLTableSchema(
        table_name="billing_disputes",
        context_str=(
            "Contains customer billing disputes including "
            "dispute reasons, dispute status (values: 'OPEN', 'RESOLVED', 'REJECTED' in uppercase), opened dates, "
            "and resolution information."
        ),
    ),

    SQLTableSchema(
        table_name="past_conversations",
        context_str=(
            "Contains logs of past chat conversations between customers and the assistant, "
            "including thread_id, role (user or assistant), content (message text), and timestamp."
        ),
    ),
]

# ----------------------------------------
# Object Index over Table Schemas
# ----------------------------------------
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
    embed_model=embed_model,
)

# ----------------------------------------
# Table Retriever
# similarity_top_k = 2
# ----------------------------------------
table_retriever = obj_index.as_retriever(
    similarity_top_k=2
)

# ----------------------------------------
# SQL Query Engine
# ----------------------------------------
query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=table_retriever,
    llm=llm,
)

# ----------------------------------------
# Main Query Function
# ----------------------------------------
def ask_sql_question(question: str) -> str:
    """
    Accepts a natural language question
    and returns synthesized SQL answer.
    """

    response = query_engine.query(question)

    return str(response)


# ----------------------------------------
# Test Run
# ----------------------------------------
if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a SQL question (or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        answer = ask_sql_question(question)

        print("\nAnswer:")
        print(answer)
