# 📊 Excel Financial Chatbot

An AI-powered chatbot that analyzes Excel and CSV files using RAG (Retrieval-Augmented Generation) architecture with multi-file relationship analysis.

## 🌟 Features

### 🤖 AI-Powered Analysis
- **Llama 3.2 3B** integration for natural language processing
- **RAG Architecture** for context-aware responses
- **Multi-file relationship detection** using embeddings
- **Financial health scoring** and trend analysis

### 📁 Multi-Format Support
- **Excel files** (.xlsx, .xls) with multiple sheets
- **CSV files** (.csv)
- **Batch processing** of multiple files
- **Cross-file correlation analysis**

### 🔍 Advanced Analytics
- **Similarity analysis** between files using vector embeddings
- **Agno framework integration** for enhanced financial insights
- **Anomaly detection** using statistical methods
- **Interactive visualizations** and charts

### 🖥️ User Interface
- **Streamlit web interface** with modern design
- **Real-time processing** with progress indicators
- **Multi-tab layout** for organized workflow
- **Responsive design** for desktop and mobile

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│   Processing    │───▶│   Vector DB     │
│   (Excel/CSV)   │    │   & Embedding   │    │   (Milvus)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat UI       │◀───│   LLM Response  │◀───│   RAG Retrieval │
│   (Streamlit)   │    │   (Llama 3.2)   │    │   & Context     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Docker** (for Milvus)
- **Ollama** with Llama 3.2 3B model

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd excel-financial-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Milvus database:**
   ```bash
   docker-compose up -d
   ```

4. **Install Ollama and Llama 3.2 3B:**
   ```bash
   # Install Ollama (Windows)
   winget install Ollama.Ollama
   
   # Pull Llama 3.2 3B model
   ollama pull llama3.2:3b
   ```

5. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser** and go to `http://localhost:8501`

## 💻 Usage

### 1. Upload Files
- Navigate to the **"Upload Files"** tab
- Select multiple Excel or CSV files
- Click **"Process All Files & Analyze Relationships"**

### 2. View Relationships
- Check the **"Relationships"** tab to see:
  - File similarity scores
  - Cross-file correlations
  - Agno advanced analysis
  - Financial health insights

### 3. Chat Analysis
- Use the **"Chat"** tab to ask questions like:
  - "Compare my files and find relationships"
  - "What are the revenue trends across files?"
  - "Identify discrepancies between datasets"

### 4. Analytics Dashboard
- Review the **"Analytics"** tab for:
  - Data overview and statistics
  - Interactive charts and visualizations
  - Per-file analysis details

## 📋 File Structure

```
chatbot/
├── streamlit_app.py           # Main Streamlit application
├── agno_integration.py        # Agno framework integration
├── milvus_setup.py           # Milvus database setup
├── excel_to_milvus.py        # Excel processing pipeline
├── agno_app.py               # Flask API server
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Milvus Docker setup
├── test_*.py                # Test files
└── README.md                # This file
```

## 🔧 Configuration

### Milvus Database
- **Host:** localhost
- **Port:** 19530
- **Collection:** excel_vectors

### Ollama Model
- **Model:** llama3.2:3b
- **API:** localhost:11434
- **Embedding size:** 256 dimensions

### Streamlit App
- **Port:** 8501
- **Theme:** Auto-detect
- **Max file size:** 200MB per file

## 🧪 API Endpoints

### Flask API (Optional)
- `POST /ingest_excel` - Upload and process Excel files
- `POST /chat` - Chat with the AI assistant

## 🔍 Key Technologies

- **🐍 Python 3.8+** - Core programming language
- **🦙 Llama 3.2 3B** - Large language model
- **🗄️ Milvus** - Vector database for embeddings
- **🎨 Streamlit** - Web interface framework
- **📊 Pandas** - Data manipulation and analysis
- **🔢 NumPy** - Numerical computing
- **🐳 Docker** - Containerization for Milvus
- **🤖 Ollama** - Local LLM inference

## 📈 Use Cases

### Financial Analysis
- **Multi-period comparison** across financial statements
- **Budget vs. actual analysis** with variance detection
- **KPI tracking** and trend identification
- **Risk assessment** through anomaly detection

### Business Intelligence
- **Cross-departmental data analysis**
- **Operational metrics correlation**
- **Performance benchmarking**
- **Data quality assessment**

### Data Integration
- **File similarity assessment**
- **Duplicate detection** across datasets
- **Data consolidation opportunities**
- **Schema alignment analysis**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta** for Llama 3.2 model
- **Zilliz** for Milvus vector database
- **Streamlit** team for the excellent web framework
- **Ollama** for local LLM inference
- **Agno** framework for enhanced analytics

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Built with ❤️ for financial data analysis and business intelligence**