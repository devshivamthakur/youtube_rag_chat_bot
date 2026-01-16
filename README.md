
# 🎬 YouTube RAG Chat Bot

Ask questions to any YouTube video and get intelligent answers using **Retrieval-Augmented Generation (RAG)** 🚀  

🔗 **GitHub Repository:**  
https://github.com/devshivamthakur/youtube_rag_chat_bot  

🎥 **Demo Video:**  
https://youtu.be/vzXoYPCco7Q  

---

## ✨ Features

- Enter any YouTube video ID  
- Automatically fetch transcript  
- Multi-language support (auto-translate to English)  
- Chunking & embeddings generation  
- Hybrid retrieval (FAISS + BM25)  
- RAG-based question answering  
- Fast responses with caching  
- Clean Streamlit UI  
- Dark mode friendly  

---

## 🧠 How It Works

1. User enters a YouTube video ID  
2. Transcript is fetched automatically  
3. Text is split into chunks  
4. Embeddings are generated  
5. Hybrid retriever searches relevant context  
6. LLM generates answer using retrieved chunks  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- LangChain  
- FAISS  
- BM25 Retriever  
- HuggingFace LLMs  
- YouTube Transcript API  
- dotenv  

---

## 📂 Project Structure

```
youtube_rag_chat_bot/
│
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/devshivamthakur/youtube_rag_chat_bot.git
cd youtube_rag_chat_bot
pip install -r requirements.txt
```

---

## 🔑 Setup Environment

Create a `.env` file:

```bash
HUGGING_FACE_TOKEN=your_token_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Paste YouTube Video ID  
2. Wait for transcript processing  
3. Ask any question  
4. Get AI-powered answer  

---

## 📌 Example Use Cases

- Learning from long tutorials  
- Podcast summarization  
- Research assistance  
- Lecture Q&A  
- Content analysis  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

## 👨‍💻 Author

**Shivam Thakur**  
🔗 https://github.com/devshivamthakur

---

## 📜 License

MIT License
