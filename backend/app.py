import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Reverting back to a single global vector store
vector_store = None
llm = None
embeddings = None
text_splitter = None

# Initialize models
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not found in environment variables!")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.15)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("✓ Models successfully initialized.")
except Exception as e:
    print(f"✗ Model initialization error: {e}")

SYSTEM_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is in the context, be detailed. 
If NOT in the context, state "The document doesn't mention this," then provide a general helpful answer.

CONTEXT:
{context}
"""

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online"}), 200

@app.route("/process_webpage", methods=["POST"])
def process_webpage():
    global vector_store
    try:
        data = request.json
        content = data.get("content", "")

        if not content or len(content.strip()) < 50:
            return jsonify({"error": "Content too short or empty"}), 400
        
        texts = text_splitter.split_text(content)
        vector_store = FAISS.from_texts(texts, embeddings)
        
        return jsonify({"status": "ready", "chunks": len(texts)})
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    global vector_store
    try:
        data = request.json or {}
        video_id = data.get("videoId", "")
        
        if not video_id:
            return jsonify({"error": "No YouTube Video ID provided"}), 400

        try:
            # SIMPLIFIED: Using get_transcript instead of list_transcripts.
            # We pass a list of languages so it tries English first, then falls back to others if needed.
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB', 'hi', 'es'])
            
            raw_text = " ".join([t['text'] for t in transcript_data])
            
            texts = text_splitter.split_text(raw_text)
            vector_store = FAISS.from_texts(texts, embeddings)
            
            return jsonify({"status": "ready", "chunks": len(texts)})
            
        except TranscriptsDisabled:
            return jsonify({"error": "Subtitles are disabled for this video."}), 400
        except NoTranscriptFound:
            return jsonify({"error": "No English or supported subtitles found for this video."}), 400
        except Exception as transcript_error:
            return jsonify({"error": f"Failed to fetch transcript: {str(transcript_error)}"}), 400

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    global vector_store
    try:
        data = request.json
        pdf_url = data.get("pdf_url")

        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(pdf_url, timeout=30, headers=headers)
        resp.raise_for_status()
        
        reader = PdfReader(io.BytesIO(resp.content))
        pdf_text = "".join([page.extract_text() or "" for page in reader.pages])
        
        texts = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(texts, embeddings)
        
        return jsonify({"status": "ready", "chunks": len(texts)})
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    global vector_store
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not vector_store:
            return jsonify({"error": "No content processed yet. Please click 'Process This Page' first."}), 400
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        
        context = "\n\n".join([d.page_content for d in docs])
        prompt = SYSTEM_TEMPLATE.replace("{context}", context) + f"\n\nUser Question: {question}"
        
        response = llm.invoke(prompt)
        return jsonify({"answer": response.content})
        
    except Exception as e:
        return jsonify({"error": f"Failed to answer: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)