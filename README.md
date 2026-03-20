# RAG Document Query System

A Retrieval-Augmented Generation (RAG) system that allows users to query company documents using natural language. Built with LangChain, ChromaDB, OpenRouter, and Streamlit.

## Features

- **Document Ingestion**: Load and process text documents from multiple companies
- **Vector Search**: Semantic search using embeddings
- **Web Interface**: Clean Streamlit UI for easy querying
- **Multi-Company Support**: Currently includes Google, Microsoft, NVIDIA, SpaceX, and Tesla documents

## Setup

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd rag-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

5. Run document ingestion:
```bash
python main.py
```

6. Run the web app:
```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Add secrets in the app settings:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
7. Deploy!

## Usage

- Enter natural language questions about the companies
- Get relevant document excerpts with source attribution
- Examples:
  - "How much did Microsoft pay for GitHub?"
  - "What was NVIDIA's first graphics card?"
  - "Who founded SpaceX?"

## Project Structure

```
├── app.py                 # Streamlit web interface
├── main.py               # Document ingestion pipeline
├── retrival_pipeline.py  # CLI retrieval interface
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (not in repo)
├── docs/                # Source documents
├── db/                  # Vector database (generated)
└── rag/                 # Virtual environment (not in repo)
```

## Technologies Used

- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **OpenRouter**: AI API provider
- **Streamlit**: Web interface
- **OpenAI Embeddings**: Text embeddings

## License

MIT License