import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.callbacks.base import BaseCallbackHandler

base_url = os.getenv("base_url")
chat_model = os.getenv("chat_model")
embedding_model = os.getenv("embedding_model")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(
            f'<div class="response-box">{self.text}</div>',
            unsafe_allow_html=True
        )
# ================= ENV ==================
load_dotenv()

# ================= SESSION ==================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.bm25 = None
    st.session_state.transcript = None

# ================= PAGE ==================
st.set_page_config(
    page_title="YouTube Video Q&A Assistant",
    page_icon="🎬",
    layout="wide"
)

# ================= CSS ==================
st.markdown("""
<style>
.main-header {font-size:2.5rem;color:#FF4B4B;text-align:center}
.response-box {background:#000;padding:20px;border-radius:10px;border-left:5px solid #FF4B4B, color:#000;font-size:1.1rem}
.warning-box {background:#fff3cd;color:#664d03;padding:15px;border-radius:10px;border-left:5px solid #ffc107}

@media (prefers-color-scheme: dark) {
.warning-box {background:#3b2f00;color:#ffd966;border-left:5px solid #ffcc00;}
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER ==================
st.markdown('<div class="main-header">🎬 YouTube Video Q&A Assistant</div>', unsafe_allow_html=True)

# ================= HELPERS ==================

def translate_to_english(text, llm):
    prompt = f"""
Translate the following text into English.
Return ONLY the translated text.

text to translate: {text}
"""
    return llm.invoke(prompt)

@st.cache_data(show_spinner=False)
def load_transcript(video_id):
    try:
        yt = YouTubeTranscriptApi()
        trans_list = yt.list(video_id)

        default = next(t for t in trans_list if t.is_generated)

        is_english = any(t.language_code == "en" for t in trans_list)

        transcript_obj = (
            next(t for t in trans_list if t.language_code == "en")
            if is_english
            else default
        )

        fetched = yt.fetch(video_id, languages=[transcript_obj.language_code])
        transcript_text = " ".join(x.text for x in fetched)

        return transcript_text
    except TranscriptsDisabled:
        st.warning("""
        <div class="warning-box">
        ⚠️ Transcripts are disabled for this video. Please enable captions on YouTube or try another video.
        </div>
        """, unsafe_allow_html=True)
        return ""
    except Exception as e:  
        print(f"Error loading transcript: {e}", flush=True)
        st.error(f"❌ Error loading transcript")
        return ""

@st.cache_resource(show_spinner=False)
def build_vector_store(transcript, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([transcript])

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id=os.getenv("embedding_model"),
        huggingfacehub_api_token=os.getenv("HUGGING_FACE_TOKEN")
    )

    faiss = FAISS.from_documents(docs, embeddings)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4

    return faiss, bm25


def process_video():
    vid = st.session_state.video_id.strip()
    if not vid:
        return

    with st.spinner("📥 Loading transcript..."):
        st.session_state.transcript = load_transcript(vid)

    with st.spinner("🔤 Generating embeddings..."):
        faiss, bm25 = build_vector_store(
            st.session_state.transcript,
            chunk_size=1000,
            chunk_overlap=200,
        )

        st.session_state.vector_store = faiss
        st.session_state.bm25 = bm25


# ================= SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.text_input(
        "YouTube Video ID:",
        key="video_id",
        value="",
        on_change=process_video   # 🔥 AUTO PROCESS
    )

# ================= MAIN ==================
col1, col2 = st.columns(2)

with col1:
    st.header("📺 Video")
    vid = st.session_state.get("video_id")
    if vid:
        st.image(f"http://img.youtube.com/vi/{vid}/0.jpg")
        st.markdown(f"[Watch Video](https://www.youtube.com/watch?v={vid})")

with col2:
    st.header("❓ Ask")

    question = st.text_area(
        "Ask a question:",
        value="What is this video about?"
    )

    process_btn = st.button(
        "🚀 Ask",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.vector_store
    )

# ================= QA ==================
if process_btn:

    faiss = st.session_state.vector_store
    bm25 = st.session_state.bm25

    retriever = EnsembleRetriever(
        retrievers=[
            faiss.as_retriever(search_type="mmr", search_kwargs={"k":6}),
            bm25
        ],
        weights=[0.7, 0.3]
    )
    response_box = st.empty()


    llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url=base_url,
    model=chat_model,
    temperature=0.2,
    streaming=True,
    callbacks=[StreamHandler(response_box)]
)

    prompt = PromptTemplate(
        template="""
Answer only from transcript.
If unsure say "I don't know".

transcript:
{context}

Question: {question}
Answer:""",
        input_variables=["context","question"]
    )

   # 7. Chain construction
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    paralle_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
   })
    main_chain = paralle_chain | prompt | llm | StrOutputParser()

    response_container = st.container()
    st.header("🎯 Answer") 
    full_response = ""

    with st.spinner("💡 Generating answer..."):
        for chunk in main_chain.stream(question):
            if "result" in chunk:
                full_response += chunk["result"]
                response_box.markdown(
                    f'<div class="response-box">{full_response}</div>',
                    unsafe_allow_html=True
                )

    with st.expander("📄 Transcript Preview"):
        st.text_area(
            "Transcript",
            st.session_state.transcript[:1000],
            height=200
        )

# ================= FOOTER ==================
st.markdown("---")
st.markdown("Built with ❤️ Streamlit + LangChain")
