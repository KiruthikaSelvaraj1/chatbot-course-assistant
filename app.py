from flask import Flask, request, jsonify, render_template_string
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.prompts import PromptTemplate
import faiss
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
LLM_MODEL = "tinyllama"
NOTES_FILE = "MACHINE LEARNING(R17A0534).pdf"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # More reliable model
EMBEDDING_DIM = 384

# Force CPU mode and reduce memory usage
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ["OLLAMA_GPU_LAYERS"] = "0"
os.environ["OLLAMA_NUM_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OLLAMA_MODELS"] = "C:\\Users\\Kiruthika\\.ollama\\models"
if not os.path.exists(NOTES_FILE):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        f.write("Course Notes - Sample\n\nNLP: Natural Language Processing enables machines to understand text.\n")

app = Flask(__name__)
query_engine = None
initialization_success = False

RESPONSE_PROMPT = PromptTemplate(
    "You are a helpful teaching assistant. Using ONLY the provided course notes, answer the question.\n"
    "Context: {context_str}\n"
    "Question: {query_str}\n"
    "Answer: Let me explain this concept from your course notes: "
)

def initialize_rag_engine():
    global query_engine, initialization_success
    try:
        logging.info(f"Loading LLM: {LLM_MODEL} and building RAG engine...")

        # 1) Create Ollama LLM client with reduced context
        llm = Ollama(
            model=LLM_MODEL, 
            request_timeout=120.0,
            num_gpu=0,
            num_ctx=512  # Reduced context window to save memory
        )

        # 2) Embeddings with offline fallback
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL,
                cache_folder="./models",  # Cache locally
                trust_remote_code=True
            )
        except Exception as e:
            logging.error(f"Failed to load embeddings: {e}")
            raise

        # 3) Load documents
        documents = SimpleDirectoryReader(input_files=[NOTES_FILE]).load_data()

        # 4) FAISS vector store
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # 5) Build index
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=embed_model
        )

        # 6) Create query engine with custom prompt
        query_engine = index.as_query_engine(
            llm=llm,
            text_qa_template=RESPONSE_PROMPT,
            response_mode="compact"
        )

        logging.info("RAG Engine successfully loaded.")
        initialization_success = True
        return True

    except Exception as e:
        logging.error(f"Failed to initialize RAG engine. Check Ollama and model status. Error: {e}")
        query_engine = None
        initialization_success = False
        return False

# Initialize at import/run
initialize_rag_engine()

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Course Notes AI Assistant</title>
    <style>
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        #chat-container {
            width: 95%;
            max-width: 900px;
            height: 90vh;
            background: #fff;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.5em;
        }
        #messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            line-height: 1.5;
            font-size: 15px;
            animation: fadeIn 0.3s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user {
            background: #007aff;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .bot {
            background: white;
            border: 1px solid #e1e4e8;
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        .error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        #upload-area {
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e1e4e8;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #file-input {
            display: none;
        }
        .upload-label {
            background: #4f46e5;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .upload-label:hover {
            background: #4338ca;
        }
        #upload-status {
            color: #4b5563;
            font-size: 14px;
        }
        #input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e4e8;
            display: flex;
            gap: 10px;
        }
        #message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e1e4e8;
            border-radius: 10px;
            font-size: 15px;
            transition: border-color 0.2s;
        }
        #message-input:focus {
            outline: none;
            border-color: #2563eb;
        }
        #send-btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 10px;
            cursor: pointer;
            transition: background 0.2s;
            font-weight: 500;
        }
        #send-btn:hover {
            background: #1d4ed8;
        }
        #send-btn:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }
        .file-info {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 8px;
            background: #f3f4f6;
            border-radius: 6px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="header">Course Notes AI Assistant</div>
        <div id="messages">
            <div class="message bot">👋 Hello! Upload your course notes (PDF) and I'll help you understand them better.</div>
            {% if not initialization_success %}
            <div class="message error">⚠️ System Error: Check if Ollama is running with model '{{ LLM_MODEL }}'</div>
            {% endif %}
        </div>
        <div id="upload-area">
            <label class="upload-label" for="file-input">
                📄 Upload PDF Notes
            </label>
            <input type="file" id="file-input" accept=".pdf" multiple />
            <span id="upload-status"></span>
        </div>
        <div id="input-area">
            <input id="message-input" placeholder="Ask any question about your notes..." />
            <button id="send-btn">Send</button>
        </div>
    </div>
    <script>
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const fileInput = document.getElementById('file-input');
        const uploadStatus = document.getElementById('upload-status');

        function addMessage(text, cls, isError=false) {
            const el = document.createElement('div');
            el.className = 'message ' + cls + (isError ? ' error' : '');
            el.textContent = text;
            messagesDiv.appendChild(el);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        fileInput.addEventListener('change', async () => {
            const files = fileInput.files;
            if (files.length === 0) return;

            uploadStatus.textContent = '📤 Uploading...';
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            try {
                const res = await fetch('/upload', { 
                    method: 'POST', 
                    body: formData 
                });
                const data = await res.json();
                
                if (res.ok) {
                    uploadStatus.textContent = '✅ Notes ready!';
                    addMessage('📚 Notes uploaded and processed successfully! Ask me anything about them.', 'bot');
                } else {
                    uploadStatus.textContent = '❌ Upload failed';
                    addMessage('Error: ' + (data.error || 'Upload failed'), 'bot', true);
                }
            } catch (e) {
                uploadStatus.textContent = '❌ Upload error';
                addMessage('Upload error - check file format', 'bot', true);
            }
        });

        async function send() {
            const msg = input.value.trim();
            if (!msg) return;
            
            addMessage(msg, 'user');
            input.value = '';
            sendBtn.disabled = true;
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                
                const data = await res.json();
                if (res.ok) {
                    addMessage(data.response, 'bot');
                } else {
                    addMessage('Error: ' + (data.error || 'Something went wrong'), 'bot', true);
                }
            } catch (e) {
                addMessage('Network error - check if server is running', 'bot', true);
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }

        sendBtn.onclick = send;
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });

        // Focus input on load
        input.focus();
    </script>
</body>
</html>
"""

# --- Routes ---
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, initialization_success=initialization_success, LLM_MODEL=LLM_MODEL)

@app.route('/upload', methods=['POST'])
def upload():
    global query_engine, initialization_success
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error':'No files provided'}), 400
    try:
        file_paths = []
        for file in files:
            if file.filename == '':
                continue
            file_path = os.path.join(os.getcwd(), file.filename)
            file.save(file_path)
            file_paths.append(file_path)
        if file_paths:
            # Reinitialize with new files
            documents = SimpleDirectoryReader(input_files=file_paths).load_data()
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
            faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store,
                embed_model=embed_model
            )
            llm = Ollama(model=LLM_MODEL, request_timeout=120.0, num_gpu=0, num_ctx=2048)
            query_engine = index.as_query_engine(
                llm=llm,
                text_qa_template=RESPONSE_PROMPT,
                response_mode="compact"
            )
            initialization_success = True
        return jsonify({'message':'Files uploaded and indexed successfully'}), 200
    except Exception as e:
        logging.error(f"Error during upload: {e}")
        return jsonify({'error':'Upload failed'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if query_engine is None:
        logging.error("Query engine is None - initialization failed")
        return jsonify({'error': f"Chatbot engine failed to initialize. Ollama model '{LLM_MODEL}' not running or insufficient memory."}), 500
    
    data = request.get_json() or {}
    user_message = data.get('message','').strip()
    
    if not user_message:
        return jsonify({'error':'No message provided'}), 400
    
    try:
        logging.info(f"Sending query to LLM: {user_message}")
        response = query_engine.query(user_message)
        text = getattr(response, "response", None) or getattr(response, "text", None) or str(response)
        logging.info(f"Got response from LLM: {text[:100]}...")
        return jsonify({'response': text}), 200
        
    except Exception as e:
        logging.error(f"Detailed error during query: {str(e)}", exc_info=True)
        return jsonify({'error':f'Internal query error: {str(e)}'}), 500

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True)