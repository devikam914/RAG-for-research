# RAG for research assistance

A simple Streamlit app for asking questions about uploaded research papers using a Retrieval-Augmented Generation (RAG) workflow.

## What it does

- Upload one or more PDF papers
- Extract text from each PDF
- Split the text into chunks and generate embeddings
- Store the vectors in a local ChromaDB database
- Retrieve the most relevant passages for a query
- Use Gemini to answer questions grounded in the paper content
- Optionally allow web-grounded answers when extra context is needed

## Tech stack

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- pypdf
- NumPy

## Project structure

```text
.
├── app.py                  # Streamlit web app
├── README.md              # Project documentation
├── overview.md            # Additional architecture notes
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── client.py          # Gemini client setup
│   ├── config.py          # App settings and environment variables
│   ├── document_processor.py  # PDF extraction and chunking
│   ├── embeddings.py      # Embedding generation
│   ├── generator.py       # LLM answer generation
│   └── vector_store.py    # ChromaDB + retrieval logic
├── chroma_db/             # Local vector store created at runtime
└── .env                   # Optional local environment variables
```

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

## How to use it

1. Upload PDF files in the sidebar.
2. Click "Build / rebuild index".
3. Ask a question in the chat box.
4. The app searches the indexed paper content and generates an answer based on the most relevant passages.

## Notes

- The app persists its vector database locally in the `chroma_db` folder.
- You can reset the index from the sidebar when you want to start over.
- The optional web toggle allows Google Search as a supplement to paper-only grounding.
