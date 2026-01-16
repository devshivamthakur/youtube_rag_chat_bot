# YouTube RAG Chat Bot

A retrieval-augmented chat bot that answers questions using YouTube video transcripts. This project extracts transcripts from YouTube videos, indexes them into a vector store, and uses a large language model (LLM) to provide context-aware answers in a conversational interface.

## Features
- Download and parse YouTube transcripts
- Create embeddings and build a vector index (Chroma / FAISS / Pinecone compatible)
- Retrieval-Augmented Generation (RAG) for accurate, context-aware answers
- Simple CLI and/or web-based chat interface
- Extensible to different LLM and vector-store providers

## Quick repository description
A small Python project that builds a RAG-powered chat bot using YouTube video transcripts and modern embedding/vector stores.

## Requirements
- Python 3.8+
- pip

## Recommended Python packages (example)
- openai (or your LLM SDK)
- youtube-transcript-api (or pytube for captions)
- langchain (optional)
- chromadb / faiss / pinecone-client
- fastapi + uvicorn (for web UI)
- python-dotenv

## Installation
1. Clone the repository
   git clone https://github.com/devshivamthakur/youtube_rag_chat_bot.git
   cd youtube_rag_chat_bot

2. Create a virtual environment and install dependencies
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

## Configuration
Create a .env file in the project root with the following environment variables (examples):
- OPENAI_API_KEY=sk-...
- YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY    # optional if using public transcripts
- PINECONE_API_KEY=...                     # optional if using Pinecone
- PINECONE_ENV=...
- CHROMA_DB_DIR=./chroma_db                # if using local Chroma

## Usage

1) Ingest a YouTube video transcript and build the index
   python scripts/ingest.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \
     --index-dir ./data/index

This script should:
- Download or fetch the transcript
- Split into chunks
- Create embeddings for each chunk
- Persist them into the configured vector store

2) Run the chat server (example using FastAPI)
   uvicorn app.main:app --reload --port 8000

3) Chat with the bot (example CLI)
   python scripts/chat_cli.py --index ./data/index

## Example commands
- Ingest multiple videos from a playlist
- Rebuild index: python scripts/ingest.py --rebuild
- Query in CLI: python scripts/chat_cli.py --query "What is the main idea of the video?"

## Project layout (suggested)
- app/                 # FastAPI app or web UI
- scripts/
  - ingest.py          # scripts to download transcripts + populate vector store
  - chat_cli.py        # minimal command-line chat interface for testing
- src/                 # library code (transcripts, indexing, embedding, retrieval)
- requirements.txt
- README.md

## Design notes / recommended libraries
- Transcripts: youtube-transcript-api, pytube, or the YouTube Data API
- Embeddings & LLM: OpenAI embeddings + chat completions, or other providers
- Vector DBs: Chroma (local), FAISS (local), Pinecone (managed)
- Use a text splitter (overlapping windows) to chunk transcripts for better retrieval
- Cache transcripts locally to avoid repeated downloads

## Environment & security
- Keep API keys out of source control; use .env or secret managers
- Avoid logging full transcripts or PII in production logs

## Development
- Run tests: pytest
- Format: black, isort
- Lint: flake8

## Contributing
Contributions are welcome. Please open an issue describing the feature or fix and submit a PR with tests where appropriate.

## License
MIT License — replace with your preferred license.

## Acknowledgements
- Inspired by retrieval-augmented systems and LangChain-style pipelines
- Thanks to maintainers of open-source libraries for embeddings and vector DBs

## Contact
For questions or help, open an issue or contact the repository owner: https://github.com/devshivamthakur