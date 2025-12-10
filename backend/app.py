import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# LangChain pieces used only for vectorstore/embeddings + text split
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Simple message wrappers (kept for chat-history conversion)
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- App init ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Globals -----------------
vector_store = None
llm = None
embeddings = None
text_splitter = None

# ---------------- Model init ----------------
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.15)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("Models initialized")
except Exception as e:
    print("Model initialization error:", e)


# ---------------- System prompt ----------------
SYSTEM_TEMPLATE = """You are a helpful assistant that answers user questions using ONLY the provided document context.
If the answer is present in the context, answer based on the context and include citations if possible.
If the answer is NOT present in the context, reply: "The document does not mention this." and then optionally give a short helpful guess.
Be concise and use bullet points when appropriate.

CONTEXT:
{context}
"""


# ---------------- Helpers ----------------
def get_english_transcript(transcript_text: str) -> str:
    """Detect & translate transcript to English using the LLM if needed."""
    if not llm or not transcript_text:
        return transcript_text
    try:
        detect_prompt = f"Detect the language of the following text and return the ISO 639-1 code only: '''{transcript_text[:400]}'''"
        lang_code = llm.invoke(detect_prompt).content.strip().lower()
        if "en" in lang_code:
            return transcript_text
        translate_prompt = f"Translate the following text to English:\n\n'''{transcript_text}'''"
        return llm.invoke(translate_prompt).content
    except Exception as e:
        print("Translation fallback error:", e)
        return transcript_text


def build_context_from_docs(docs, max_chars=3000):
    """Combine retrieved docs into a single context string (truncate to max_chars)."""
    parts = []
    total = 0
    for d in docs:
        # doc may be a dict-like or object with page_content
        text = None
        if hasattr(d, "page_content"):
            text = d.page_content
        elif isinstance(d, dict) and "page_content" in d:
            text = d["page_content"]
        else:
            text = str(d)
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            parts.append(text[:remaining])
            total += remaining
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def call_llm_with_context(question: str, context: str, history=None) -> str:
    """
    Compose the final prompt and call the LLM.
    history is optional list of HumanMessage/AIMessage converted strings if you want continuity.
    """
    # short guard
    if not llm:
        return "Models are not initialized."

    # Compose prompt: include system prompt with context + question
    prompt = SYSTEM_TEMPLATE.replace("{context}", context if context else "No context available.")
    prompt += "\n\nUser question:\n" + question

    # Optionally include short history at the end (if provided)
    if history:
        hist_texts = []
        for item in history:
            t = item.content if hasattr(item, "content") else str(item)
            hist_texts.append(t)
        if hist_texts:
            prompt += "\n\nConversation history (most recent last):\n" + "\n".join(hist_texts[-6:])

    try:
        resp = llm.invoke(prompt)
        # resp may be an object with .content
        return getattr(resp, "content", str(resp))
    except Exception as e:
        print("LLM invocation error:", e)
        return f"LLM error: {e}"


# ---------------- Health ----------------
@app.route("/", methods=["GET"])
def health():
    return "OK", 200


# ---------------- Process webpage ----------------
@app.route("/process_webpage", methods=["POST"])
def process_webpage():
    global vector_store
    if not embeddings or not text_splitter:
        return jsonify({"error": "Models not initialized"}), 500

    try:
        data = request.json or {}
        page_content = data.get("content", "")
        if not page_content:
            return jsonify({"error": "No content provided"}), 400

        soup = BeautifulSoup(page_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        if not text.strip():
            return jsonify({"error": "No textual content found"}), 400

        texts = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready", "message": f"Processed {len(text)} characters."})
    except Exception as e:
        print("process_webpage error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- Process YouTube ----------------
@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    global vector_store
    if not embeddings or not text_splitter:
        return jsonify({"error": "Models not initialized"}), 500

    try:
        data = request.json or {}
        video_id = data.get("videoId", "")
        if not video_id:
            return jsonify({"error": "No YouTube Video ID provided"}), 400

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi", "es", "de"])
            raw_transcript = " ".join([item.get("text", "") for item in transcript_list])
        except Exception as e:
            print("YouTube transcript unavailable:", e)
            # AI fallback summary if LLM available
            if llm:
                fallback_prompt = (
                    f"Captions for the YouTube video https://www.youtube.com/watch?v={video_id} are unavailable. "
                    "Based on the video link, provide a concise pseudo-transcript summarizing likely spoken content "
                    "and main points in bullet form."
                )
                try:
                    raw_transcript = llm.invoke(fallback_prompt).content
                except Exception as ee:
                    print("AI fallback failed:", ee)
                    return jsonify({"status": "no_transcript", "message": "Transcript unavailable"}), 200
            else:
                return jsonify({"status": "no_transcript", "message": "Transcript unavailable"}), 200

        english_transcript = get_english_transcript(raw_transcript)
        texts = text_splitter.split_text(english_transcript)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready", "message": f"Processed {len(english_transcript)} characters."})
    except Exception as e:
        print("process_youtube error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- Process PDF ----------------
@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    global vector_store
    if not embeddings or not text_splitter:
        return jsonify({"error": "Models not initialized"}), 500

    try:
        data = request.json or {}
        pdf_url = data.get("pdf_url", "")
        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400

        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        pdf_text = ""
        for p in reader.pages:
            text = p.extract_text() or ""
            pdf_text += text + "\n"

        if not pdf_text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        texts = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready", "message": f"Processed {len(pdf_text)} characters."})
    except Exception as e:
        print("process_pdf error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- Ask endpoint (manual retrieval + LLM) ----------------
@app.route("/ask", methods=["POST"])
def ask_question():
    global vector_store, llm
    if vector_store is None:
        return jsonify({"error": "Page not processed yet."}), 400
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        history_list = data.get("history", [])

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # convert history to Human/AI message objects if present (optional)
        history_msgs = []
        for item in history_list:
            if item.get("type") == "human":
                history_msgs.append(HumanMessage(content=item.get("content")))
            elif item.get("type") == "ai":
                history_msgs.append(AIMessage(content=item.get("content")))

        # Retrieve top-k similar docs using FAISS vector store
        try:
            # similarity_search is available on LangChain FAISS vectorstore
            docs = vector_store.similarity_search(question, k=4)
        except Exception:
            # fallback - try as_retriever then .get_relevant_documents
            try:
                retr = vector_store.as_retriever()
                docs = retr.get_relevant_documents(question)
            except Exception as e:
                print("Retrieval error:", e)
                docs = []

        context = build_context_from_docs(docs, max_chars=3500)

        # Compose and call LLM using the context
        answer = call_llm_with_context(question, context, history=history_msgs)

        return jsonify({"answer": answer})
    except Exception as e:
        print("ask error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- Run (never used on Render, kept for local debug) ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
