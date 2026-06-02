import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)
# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "data", "documents")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_index")

# -----------------------------
# Configure Local Embedding Model
# -----------------------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load or Build Index
# -----------------------------
def get_index():
    """
    Loads existing vector index if available.
    Otherwise builds a new one and persists it.
    """

    index_file = os.path.join(
        VECTOR_STORE_PATH,
        "docstore.json"
    )

    # Check if actual persisted index exists
    if os.path.exists(index_file):

        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH
        )

        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model,
        )

        print("Loading existing vector index.")

    else:
        print("Index not found. Building a new one...")
        documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=VECTOR_STORE_PATH)
        print("Index built and persisted.")

    return index
# -----------------------------
# Query Function
# -----------------------------
def ask_question(question: str) -> str:


    index = get_index()


    query_engine = index.as_query_engine(llm=llm)


    response = query_engine.query(question)

    return str(response)


# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":

    while True:
        user_question = input("\nAsk a question (or type 'exit'): ")

        if user_question.lower() == "exit":
            break

        answer = ask_question(user_question)

        print("\nAnswer:")
        print(answer)