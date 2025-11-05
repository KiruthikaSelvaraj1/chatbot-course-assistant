# Course Notes AI Assistant 🎓

> An intelligent chatbot leveraging RAG (Retrieval-Augmented Generation) technology with Ollama-TinyLlama to transform course materials into interactive learning experiences.

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Framework](https://img.shields.io/badge/framework-Flask-orange.svg)

## 🌟 Overview

This project implements a local-first RAG system that enables students to:
- Upload course materials in PDF format
- Ask questions in natural language
- Receive contextually accurate answers
- Study efficiently with AI assistance
- Maintain complete privacy of their materials

## ✨ Key Features

### Core Capabilities
- 📚 PDF Document Processing
- 💡 Context-Aware Responses
- 🔍 Semantic Search
- 🤖 Local LLM Integration
- 🔒 Privacy-First Architecture

### Technical Features
- 🚀 Memory-Optimized RAG Pipeline
- 💻 Modern Web Interface
- 📱 Responsive Design
- ⚡ Fast Vector Search
- 🔄 Real-Time Processing

## 🛠️ Technology Stack

### Backend Infrastructure
- **Web Framework**: Flask
- **RAG Engine**: LlamaIndex
- **LLM Host**: Ollama
- **Vector DB**: FAISS
- **Embeddings**: HuggingFace

### AI Models
- **Language Model**: TinyLlama
- **Embedding Model**: all-mpnet-base-v2
- **Vector Dimension**: 384D

### Frontend Technologies
- **HTML5**: Structure
- **CSS3**: Styling
- **JavaScript**: Interactivity
- **Fetch API**: Async Communication

## 📋 System Requirements

### Hardware
- **CPU**: Multi-core processor
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for initial setup

### Software
- **OS**: Windows 10/11
- **Python**: Version 3.11+
- **Ollama**: Latest version
- **Browser**: Modern web browser

## ⚡ Installation Guide

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/yourusername/course-chatbot
cd course-chatbot

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup
```powershell
# Install Ollama from https://ollama.ai/
# Pull required model
ollama pull tinyllama
```

### 3. Application Launch
```powershell
# Terminal 1: Start Ollama
$env:CUDA_VISIBLE_DEVICES = ""
ollama serve

# Terminal 2: Start Flask
python app.py
```

## 💻 Usage Instructions

1. **Access Application**
   - Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Modern browser recommended

2. **Upload Materials**
   - Click "Upload PDF Notes"
   - Select course materials
   - Wait for processing

3. **Ask Questions**
   - Type your question
   - Press Enter or click Send
   - View AI-generated response

## 🔧 Configuration Options

### Model Settings
```python
# CPU Optimization
LLM_MODEL = "tinyllama"
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 384
CONTEXT_WINDOW = 512

# Environment Variables
CUDA_VISIBLE_DEVICES = ""
OLLAMA_HOST = "http://127.0.0.1:11434"
```

## 📊 Performance Metrics

- **Startup Time**: ~5-10 seconds
- **Document Processing**: ~2-3 seconds/page
- **Response Time**: 2-5 seconds
- **Memory Usage**: 2-4GB RAM
- **Storage Required**: ~1GB

## ❗ Troubleshooting Guide

### Common Issues

1. **Connection Errors**
   ```powershell
   # Check Ollama status
   ollama list
   netstat -ano | findstr "11434"
   ```

2. **Memory Problems**
   ```powershell
   # Monitor resources
   Get-Process python | Select-Object CPU, WorkingSet
   ```

3. **Model Issues**
   ```powershell
   # Verify model
   ollama list
   ollama pull tinyllama --clean
   ```

## 🔒 Security Features

- **Local Processing**: No cloud dependencies
- **Data Privacy**: In-memory operations only
- **No Persistence**: Documents cleared after session
- **Isolated Environment**: Containerized execution

## 🚀 Development Roadmap

### Short Term
- [ ] Multi-document support
- [ ] Answer citations
- [ ] Session management

### Long Term
- [ ] GPU acceleration
- [ ] Custom model support
- [ ] Collaborative features

## 🤝 Contributing Guidelines

1. **Setup Development Environment**
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Development Workflow**
   - Fork repository
   - Create feature branch
   - Implement changes
   - Submit PR

## 📄 License & Credits

### License
MIT License - See [LICENSE](LICENSE)

### Acknowledgments
- LlamaIndex Team
- Ollama Project
- FAISS Developers
- Flask Community

---

**Note**: This project is under active development. For production deployment, implement additional security measures and performance optimizations.

For support: [Create an Issue]()

