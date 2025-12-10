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

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# ------------- Flask Init -------------
app = Flask(__name__)
CORS(app)

vector_store = None
llm = None
embeddings = None
text_splitter = None

# ------------- Model Setup -------------
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
except Exception as e:
    print("Model init error:", e)


# ------------- System Prompt -------------
SYSTEM_TEMPLATE = """
You are a helpful assistant.
You answer questions using ONLY the provided document context.
If an answer is not in the document, say so politely and then attempt a helpful guess.
Keep answers clear and well formatted.

Context:
{context}
"""


# ------------- Language Detection + Translation -------------
def get_english_transcript(transcript_text):
    if not llm:
        return transcript_text

    try:
        detect_prompt = (
            f"Detect language. Output only ISO code: '{transcript_text[:300]}'"
        )
        lang_code = llm.invoke(detect_prompt).content.strip().lower()

        if "en" in lang_code:
            return transcript_text

        translation_prompt = f"Translate to English: '{transcript_text}'"
        return llm.invoke(translation_prompt).content

    except Exception as e:
        print("Translation error:", e)
        return transcript_text


# ------------- Health Check -------------
@app.route("/")
def health():
    return "Backend running", 200


# ------------- Web Page Processing -------------
@app.route("/process_webpage", methods=["POST"])
def process_webpage():
    global vector_store
    try:
        data = request.json
        page_content = data.get("content", "")

        if not page_content:
            return jsonify({"error": "No content provided"}), 400

        soup = BeautifulSoup(page_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        docs = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(docs, embeddings)

        return jsonify({"status": "ready", "message": f"Processed {len(text)} characters."})

    except Exception as e:
        print("Web error:", e)
        return jsonify({"error": str(e)}), 500


# ------------- YouTube Processing -------------
@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    global vector_store

    try:
        data = request.json
        video_id = data.get("videoId", "")

        if not video_id:
            return jsonify({"error": "No YouTube Video ID provided"}), 400

        print(f"Fetching transcript for: {video_id}")

        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "hi", "es", "de"]
            )
            raw_transcript = " ".join([item["text"] for item in transcript_data])

        except Exception as e:
            print("Transcript unavailable. AI fallback activated:", e)

            fallback_prompt = f"""
            YouTube transcript was blocked/unavailable.
            Summarize the key spoken content of this video:
            https://www.youtube.com/watch?v={video_id}

            Include:
            - What the speaker talks about
            - Main points explained
            - Bullet structure
            """

            try:
                raw_transcript = llm.invoke(fallback_prompt).content
            except Exception:
                return jsonify({
                    "status": "no_transcript",
                    "message": "Transcript unavailable — AI fallback failed."
                }), 200

        english_transcript = get_english_transcript(raw_transcript)

        docs = text_splitter.split_text(english_transcript)
        vector_store = FAISS.from_texts(docs, embeddings)

        return jsonify({
            "status": "ready",
            "message": f"Processed {len(english_transcript)} characters."
        })

    except Exception as e:
        print("YouTube error:", e)
        return jsonify({"error": str(e)}), 500


# ------------- PDF Processing -------------
@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    global vector_store

    try:
        data = request.json
        pdf_url = data.get("pdf_url", "")
        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400

        response = requests.get(pdf_url)
        response.raise_for_status()

        reader = PdfReader(io.BytesIO(response.content))
        pdf_text = "".join([page.extract_text() + "\n" for page in reader.pages])

        if not pdf_text.strip():
            return jsonify({"error": "Could not read text from PDF"}), 400

        docs = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(docs, embeddings)

        return jsonify({
            "status": "ready",
            "message": f"Processed {len(pdf_text)} characters."
        })

    except Exception as e:
        print("PDF error:", e)
        return jsonify({"error": str(e)}), 500


# ------------- Ask Question -------------
@app.route("/ask", methods=["POST"])
def ask_question():
    global vector_store, llm

    if not vector_store:
        return jsonify({"error": "Page not processed yet."}), 400

    try:
        data = request.json
        question = data.get("question", "")
        history_list = data.get("history", [])

        if not question:
            return jsonify({"error": "No question provided"}), 400

        chat_history = []
        for item in history_list:
            if item.get("type") == "human":
                chat_history.append(HumanMessage(content=item.get("content")))
            elif item.get("type") == "ai":
                chat_history.append(AIMessage(content=item.get("content")))

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            combine_docs_chain_kwargs={
                "prompt": qa_prompt,
                "document_variable_name": "context"
            }
        )

        response = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        return jsonify({"answer": response.get("answer")})

    except Exception as e:
        print("Ask error:", e)
        return jsonify({"error": str(e)}), 500
