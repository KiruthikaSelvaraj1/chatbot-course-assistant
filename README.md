# chatbot-course-assistant

# Course Notes AI Assistant 🎓  Transform your course materials into an intelligent study companion. This privacy-focused chatbot uses RAG technology with TinyLlama to provide AI-powered answers from your PDF notes, all while running locally on your machine.
# Course Notes AI Assistant 🎓

Transform your course materials into an intelligent study companion. This privacy-focused chatbot uses RAG (Retrieval-Augmented Generation) technology with TinyLlama to provide AI-powered answers from your PDF notes, all while running locally on your machine.

## 🌟 Features

- **Privacy-First**: All processing happens locally - no data sent to external servers
- **RAG Technology**: Combines retrieval from your notes with AI generation for accurate answers
- **PDF Support**: Upload and process PDF course materials
- **Local LLM**: Uses TinyLlama model running via Ollama
- **Web Interface**: Clean, modern chat interface built with Flask
- **Vector Search**: FAISS-powered semantic search for relevant content
- **Real-time Chat**: Instant responses to your questions about course material

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running
- TinyLlama model downloaded via Ollama

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KiruthikaSelvaraj1/chatbot-course-assistant.git
   cd chatbot-course-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv chatbot_env
   chatbot_env\Scripts\activate  # Windows
   # source chatbot_env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   ```bash
   # Download and install Ollama from https://ollama.ai/
   ollama pull tinyllama
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

## 📖 Usage

1. **Upload Course Notes**: Click "Upload PDF Notes" to upload your course materials
2. **Ask Questions**: Type questions about your course content in the chat interface
3. **Get Answers**: Receive AI-powered responses based on your uploaded notes

### Example Questions
- "Explain the concept of machine learning from the notes"
- "What are the key points about natural language processing?"
- "Summarize chapter 3 from my course materials"

## 🛠️ Technical Details

### Architecture

- **Frontend**: Flask web application with HTML/CSS/JavaScript
- **Backend**: Python Flask server
- **AI Engine**: LlamaIndex with TinyLlama LLM
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace transformers (all-mpnet-base-v2)

### Key Components

- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Local Processing**: All AI processing happens on your machine
- **Memory Efficient**: Optimized for CPU usage with reduced context windows
- **Error Handling**: Robust error handling for model loading and queries

### Dependencies

- `llama-index`: RAG framework
- `faiss-cpu`: Vector similarity search
- `transformers`: HuggingFace models
- `torch`: PyTorch for ML operations
- `flask`: Web framework
- `llama-index-llms-ollama`: Ollama integration
- `llama-index-embeddings-huggingface`: Embedding models

## 🔧 Configuration

### Environment Variables

```bash
OLLAMA_HOST=http://127.0.0.1:11434  # Ollama server address
OLLAMA_GPU_LAYERS=0                 # Disable GPU layers for CPU-only
OLLAMA_NUM_GPU=0                    # Number of GPU layers
CUDA_VISIBLE_DEVICES=""             # Disable CUDA
OLLAMA_MODELS=C:\Users\Kiruthika\.ollama\models  # Model storage path
```

### Model Configuration

- **LLM Model**: TinyLlama (configurable in app.py)
- **Embedding Model**: all-mpnet-base-v2 (384 dimensions)
- **Context Window**: 512 tokens (configurable)
- **Vector Dimensions**: 384

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure Ollama is running: `ollama serve`
   - Pull the model: `ollama pull tinyllama`

2. **Memory errors**
   - Reduce context window in app.py
   - Ensure sufficient RAM (4GB+ recommended)

3. **Upload fails**
   - Check file format (PDF only)
   - Ensure write permissions in the directory

4. **Slow responses**
   - Model is running locally - first response may take longer
   - Subsequent queries are faster due to caching

### Logs

Check the console output for detailed error messages and debugging information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM serving
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the efficient LLM

---

**Made with ❤️ for students and educators**
