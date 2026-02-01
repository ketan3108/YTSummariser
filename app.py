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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal NoteGPT", layout="wide", page_icon="📝")

st.title("📝 Universal NoteGPT (Deep Dive Edition)")
st.markdown("Extracts detailed notes from **ANY** video using **Smart Fallback** (API → yt-dlp).")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Try to load key from Secrets (Back-end)
    if "GROQ_API_KEY" in st.secrets:
        st.success("✅ API Key loaded from secrets")
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        # If no secret found, ask the user (Fallback)
        api_key = st.text_input("Enter Groq API Key", type="password")

    # Updated Model List (Using Latest Llama 3.3/3.1)
    model_option = st.selectbox(
        "AI Model",
        [
            "llama-3.3-70b-versatile",  # Smartest (Recommended for detailed notes)
            "llama-3.1-8b-instant",  # Fastest
            "mixtral-8x7b-32768"  # Large Context
        ],
        index=0
    )

    st.info("💡 **Tip:** Use 'Llama 3.3 70B' for the most detailed summaries.")


# --- TIER 1 & 2 EXTRACTION LOGIC ---

def get_video_id(url):
    """Extracts Video ID from URL"""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None


def fetch_with_ytdlp(url, video_id):
    """
    Tier 2 Fallback: Uses yt-dlp to download the hidden VTT subtitle file.
    This bypasses the 'TranscriptsDisabled' error by finding auto-subs directly.
    """
    try:
        ydl_opts = {
            'skip_download': True,  # Don't download video (Speed)
            'writeautomaticsub': True,  # Grab auto-generated subs
            'writesubtitles': True,  # Grab manual subs if available
            'subtitleslangs': ['en'],  # Prefer English
            'outtmpl': f'temp_{video_id}',
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the .vtt file (name varies slightly)
        files = glob.glob(f"temp_{video_id}*.vtt")
        if not files:
            return None

        vtt_file = files[0]
        # Parse VTT to clean text
        caption = webvtt.read(vtt_file)
        text_content = " ".join([line.text for line in caption])

        # Clean up temp files
        for f in files:
            os.remove(f)

        return text_content
    except Exception as e:
        return None


@st.cache_data
def get_transcript(video_url, video_id):
    """
    Smart Extraction Strategy:
    1. Try YouTube API (Fastest, Pythonic).
    2. If that fails/is blocked, use yt-dlp (Robust, mimics browser).
    """
    # METHOD 1: Standard API
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to find Manual English -> Auto English
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US'])

        text_data = transcript.fetch()
        full_text = " ".join([t['text'] for t in text_data])
        return full_text, "Standard API"

    except Exception as e:
        # METHOD 2: yt-dlp Fallback
        text = fetch_with_ytdlp(video_url, video_id)
        if text:
            return text, "Robust Fallback (yt-dlp)"
        else:
            return None, "Failed (No subtitles found)"


# --- AI PROCESSING (UPDATED PROMPTS) ---

@st.cache_resource
def create_vector_db(text):
    """Creates Local Vector DB for Q&A"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def get_summary(text, client, model):
    """
    Detailed 'Chain of Density' Prompt.
    Forces the AI to be specific, not vague.
    """
    prompt = f"""
    You are an expert analyst and educational note-taker. 
    Analyze the following YouTube transcript and produce a detailed, high-value study guide.

    ### INSTRUCTIONS:
    1. **The Core Thesis**: In 2-3 sentences, define exactly what this video argues or teaches.
    2. **Key Concepts (The "Meat")**: Identify the top 5-7 most important distinct ideas. 
       - For EACH idea, write a **Bold Headline**.
       - Provide a **Detailed Explanation** (3-4 sentences).
       - **Cite Evidence**: Mention specific examples, numbers, tools, or case studies used in the video to support this point.
    3. **Actionable Steps**: What can the viewer *do* immediately after watching this? (List 3-5 concrete steps).
    4. **Jargon Buster**: Define any technical terms or specific tools mentioned.

    ### RULES:
    - Do NOT be vague (e.g., instead of "he discussed marketing", say "he discussed the 'Viral Loop' marketing strategy").
    - Use Markdown formatting.
    - Focus on unique insights, not generic fluff.

    ### TRANSCRIPT START:
    {text[:28000]}... (truncated)
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,  # Low temp = more factual/rigorous
    )
    return chat_completion.choices[0].message.content


def ask_bot(question, db, client, model):
    """Deep-Dive Q&A"""
    # Retrieve top 5 chunks (More context = Better answers)
    docs = db.similarity_search(question, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. Answer the user's question in detail using ONLY the context below.

    CONTEXT:
    {context}

    USER QUESTION: 
    {question}

    INSTRUCTIONS:
    - Provide a direct, detailed answer.
    - If the context mentions steps or a process, list them clearly.
    - If the answer is not in the context, admit it politely.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return chat_completion.choices[0].message.content


# --- MAIN APP LOGIC ---

if not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to start.")
else:
    client = Groq(api_key=api_key)

    # URL Input
    url = st.text_input("🔗 Paste YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

    if url:
        video_id = get_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            # Show Video Thumbnail
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=350)

            # Extract Transcript
            with st.spinner("⚡ Extracting Transcript (Checking API then Fallback)..."):
                transcript_text, source_method = get_transcript(url, video_id)

            if not transcript_text:
                st.error("❌ Could not retrieve transcript. The video might be completely silent or region-locked.")
            else:
                st.success(f"✅ Extracted successfully via: {source_method}")

                # Setup Tabs
                tab1, tab2 = st.tabs(["📝 Detailed Notes", "💬 Deep Chat"])

                # --- TAB 1: SUMMARY ---
                with tab1:
                    if st.button("Generate Detailed Notes"):
                        with st.spinner("🤖 Analyzing content... (This uses the 70B model for high detail)"):
                            summary = get_summary(transcript_text, client, model_option)
                            st.markdown(summary)

                # --- TAB 2: CHAT ---
                with tab2:
                    st.write("Ask specific questions about the video content.")

                    # Initialize Vector DB (Lazy Load)
                    if "vector_db" not in st.session_state:
                        with st.spinner("🧠 Indexing knowledge base..."):
                            st.session_state.vector_db = create_vector_db(transcript_text)

                    # Initialize Chat History
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display History
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Chat Input
                    if prompt := st.chat_input("Ex: What tools did they mention?"):
                        # Show user message
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Generate & Show Answer
                        with st.chat_message("assistant"):
                            with st.spinner("Searching video context..."):
                                response = ask_bot(prompt, st.session_state.vector_db, client, model_option)
                                st.markdown(response)

                        # Save Answer
                        st.session_state.messages.append({"role": "assistant", "content": response})