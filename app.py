import streamlit as st
import os
import re
import glob
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yt_dlp
import webvtt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Universal NoteGPT", layout="wide", page_icon="📊")

st.title("📊 Universal NoteGPT (Data-First Edition)")
st.markdown("Extracts **Tables, Metrics, and Structured Lists** from any video.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Auto-Load API Key from Secrets (Best Practice)
    if "GROQ_API_KEY" in st.secrets:
        st.success("✅ API Key loaded from secrets")
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        api_key = st.text_input("Enter Groq API Key", type="password")
        st.markdown("[Get Free Key](https://console.groq.com/keys)")

    # UPDATED MODEL LIST (Fixed for 2026/Current API)
    model_option = st.selectbox(
        "AI Model",
        [
            "llama-3.3-70b-versatile",  # Best for Data Extraction
            "llama-3.1-8b-instant",  # Fastest
            "mixtral-8x7b-32768"  # Good alternative
        ],
        index=0
    )


# --- ROBUST EXTRACTION (API + FALLBACK) ---

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None


def fetch_with_ytdlp(url, video_id):
    """Fallback: Downloads subtitle file directly."""
    try:
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'outtmpl': f'temp_{video_id}',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = glob.glob(f"temp_{video_id}*.vtt")
        if not files: return None

        caption = webvtt.read(files[0])
        text = " ".join([line.text for line in caption])

        for f in files: os.remove(f)
        return text
    except:
        return None


@st.cache_data
def get_transcript(video_url, video_id):
    try:
        # Method 1: API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US'])
        text = " ".join([t['text'] for t in transcript.fetch()])
        return text, "Standard API"
    except:
        # Method 2: Fallback
        text = fetch_with_ytdlp(video_url, video_id)
        return (text, "Robust Fallback") if text else (None, "Failed")


# --- AI PROCESSING (NEW DATA-HEAVY PROMPT) ---

@st.cache_resource
def create_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def get_summary(text, client, model):
    """
    DATA-FIRST PROMPT:
    Forces the AI to use Markdown Tables and Lists.
    """
    prompt = f"""
    You are a Data Analyst and Technical Writer. 
    Analyze this transcript and output a STRUCTURED REPORT.

    ### STRICT OUTPUT FORMAT (Markdown):

    **1. 📋 Executive Brief**
    * **Topic:** (1 sentence)
    * **Core Problem/Thesis:** (1-2 sentences)
    * **Target Audience:** (Who is this for?)

    **2. 📊 Key Metrics & Numbers (MANDATORY TABLE)**
    * Extract ANY metrics: Prices, Percentages, Timeframes, Steps, or Quantities.
    * Format as a Markdown Table:
    | Metric/Item | Value/Detail | Context |
    | :--- | :--- | :--- |
    | (e.g. Price) | $500 | Cost of tool mentioned |

    **3. 🛠️ Tools & Resources Mentioned**
    * List specific tools, software, books, or websites mentioned.
    * Format as a checklist:
    * [ ] Tool Name: Brief description

    **4. 💡 Key Concepts & Strategy**
    * **Concept 1 (Headline):** Explanation (2 sentences max).
    * **Concept 2 (Headline):** Explanation.

    **5. 🚀 Action Plan (Step-by-Step)**
    1.  Step 1
    2.  Step 2

    **RULES:**
    * NO long paragraphs.
    * If no numbers exist, write "No metrics found in this video."
    * Be crisp, dry, and professional.

    TRANSCRIPT START:
    {text[:28000]}...
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,  # Very low temp for strict formatting
    )
    return chat_completion.choices[0].message.content


def ask_bot(question, db, client, model):
    docs = db.similarity_search(question, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer (Be detailed and list steps if possible):"

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return chat_completion.choices[0].message.content


# --- UI ---

if not api_key:
    st.warning("⚠️ Enter Groq API Key in Sidebar or Secrets")
else:
    client = Groq(api_key=api_key)
    url = st.text_input("🔗 Paste YouTube URL")

    if url:
        video_id = get_video_id(url)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=350)

            with st.spinner("⚡ Extracting Data..."):
                transcript, source = get_transcript(url, video_id)

            if transcript:
                st.success(f"✅ Extracted via {source}")

                tab1, tab2 = st.tabs(["📊 Structured Data Report", "💬 Deep Chat"])

                with tab1:
                    if st.button("Generate Report"):
                        with st.spinner("Analyzing metrics and data points..."):
                            summary = get_summary(transcript, client, model_option)
                            st.markdown(summary)

                with tab2:
                    if "vector_db" not in st.session_state:
                        st.session_state.vector_db = create_vector_db(transcript)

                    if "messages" not in st.session_state: st.session_state.messages = []

                    for m in st.session_state.messages:
                        with st.chat_message(m["role"]): st.markdown(m["content"])

                    if q := st.chat_input("Ask about specific numbers..."):
                        st.session_state.messages.append({"role": "user", "content": q})
                        with st.chat_message("user"): st.markdown(q)

                        ans = ask_bot(q, st.session_state.vector_db, client, model_option)
                        with st.chat_message("assistant"): st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
            else:
                st.error("Failed to extract transcript.")