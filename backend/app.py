import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

vector_store = None
llm = None
embeddings = None
text_splitter = None

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.15)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("Models successfully initialized.")
except Exception as e:
    print(f"Model initialization error: {e}")

SYSTEM_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is in the context, be detailed. 
If NOT in the context, state "The document doesn't mention this," then provide a general helpful answer.

CONTEXT:
{context}
"""

def get_english_transcript(transcript_text):
    if not llm or not transcript_text: return transcript_text
    try:
        prompt = f"Translate the following to English if it's in another language. Return only the translated text: {transcript_text[:3000]}"
        return llm.invoke(prompt).content
    except:
        return transcript_text

@app.route("/", methods=["GET"])
def health():
    return "Backend is Live!", 200

@app.route("/process_webpage", methods=["POST"])
def process_webpage():
    global vector_store
    try:
        data = request.json
        soup = BeautifulSoup(data.get("content", ""), "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        texts = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    global vector_store
    try:
        video_id = request.json.get("videoId")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = next(iter(transcript_list))
        
        raw_text = " ".join([t['text'] for t in transcript.fetch()])
        english_text = get_english_transcript(raw_text)
        
        texts = text_splitter.split_text(english_text)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"error": f"YouTube Error: {str(e)}"}), 500

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    global vector_store
    try:
        pdf_url = request.json.get("pdf_url")
        resp = requests.get(pdf_url, timeout=30)
        reader = PdfReader(io.BytesIO(resp.content))
        pdf_text = "".join([p.extract_text() for p in reader.pages])
        
        texts = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(texts, embeddings)
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    global vector_store
    if not vector_store: return jsonify({"error": "Nothing processed yet"}), 400
    try:
        data = request.json
        question = data.get("question")
        docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = SYSTEM_TEMPLATE.replace("{context}", context) + f"\nUser Question: {question}"
        answer = llm.invoke(prompt).content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)