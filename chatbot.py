import os
import sys

# faiss import with helpful message
try:
    import faiss
except Exception as e:
    print("ERROR: faiss import failed. On Windows prefer: conda install -c pytorch faiss-cpu")
    raise

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.prompts import PromptTemplate

LLM_MODEL = "llama3.1"
NOTES_FILE = "MACHINE LEARNING(R17A0534).pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Add custom prompt to query engine
CUSTOM_PROMPT = PromptTemplate("""
You are a knowledgeable and friendly teaching assistant. Your task is to help students understand their course material.

RULES:
1. ONLY use information from the provided context to answer
2. If the context doesn't contain relevant information, say: "I don't find specific information about this in your course notes."
3. Keep answers clear and focused on the course material
4. Use examples from the context when available
5. Format complex concepts in a structured way

Context from course notes:
{context_str}

Student Question: {query_str}

Teaching Assistant's Response:""")

def ensure_notes():
    if not os.path.exists(NOTES_FILE):
        print(f"ERROR: Notes file '{NOTES_FILE}' not found. Create it and re-run.")
        sys.exit(1)

def build_index():
    print(f"Loading documents from {NOTES_FILE}...")
    documents = SimpleDirectoryReader(input_files=[NOTES_FILE]).load_data()

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    print("Creating FAISS index...")
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    print("Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model
    )
    return index

def create_llm():
    print(f"Creating LLM client for model: {LLM_MODEL} ...")
    try:
        llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
        return llm
    except Exception as e:
        msg = str(e).lower()
        print(f"ERROR: Failed to initialize LLM '{LLM_MODEL}': {e}")
        if any(k in msg for k in ("memory", "out of memory", "unable to load full model")):
            print("Possible fixes:")
            print(' 1) Force Ollama to CPU: stop ollama and restart with:')
            print('    $env:CUDA_VISIBLE_DEVICES = ""')
            print("    ollama serve")
            print(" 2) Free system RAM / close heavy apps or increase Windows pagefile.")
            print(" 3) Use a smaller or quantized model.")
        sys.exit(1)

def chat():
    ensure_notes()
    index = build_index()
    llm = create_llm()

    # Enhanced query engine configuration
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=CUSTOM_PROMPT,
        response_mode="tree_summarize",  # More contextual responses
        similarity_top_k=3,  # Retrieve more context
        streaming=True
    )

    print("\n" + "="*50)
    print(f"👋 Hi! I'm your Course Assistant using {LLM_MODEL}")
    print("📚 I'll help explain concepts from your course notes.")
    print("💡 Ask specific questions for better answers!")
    print("="*50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\n👋 Goodbye! Happy studying!")
            break

        try:
            print("\n🤔 Analyzing your course notes...")
            response = query_engine.query(user_input)
            text = getattr(response, "response", None) or getattr(response, "text", None) or str(response)
            print(f"\n🎓 Answer:\n{text}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Tip: Ensure Ollama is running and try asking the question differently.")

if __name__ == "__main__":
    chat()