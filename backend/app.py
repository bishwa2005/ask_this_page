import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)

vector_store = None
llm = None
embeddings = None
text_splitter = None

# Initialize models with better error handling
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not found in environment variables!")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.15)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("✓ Models successfully initialized.")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    print("Make sure GOOGLE_API_KEY is set and langchain-google-genai is installed")

SYSTEM_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is in the context, be detailed. 
If NOT in the context, state "The document doesn't mention this," then provide a general helpful answer.

CONTEXT:
{context}
"""

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "models_initialized": llm is not None and embeddings is not None,
        "api_key_set": bool(os.environ.get("GOOGLE_API_KEY"))
    }), 200

@app.route("/process_webpage", methods=["POST"])
def process_webpage():
    """Process webpage content and create vector store"""
    global vector_store
    try:
        data = request.json
        if not data or "content" not in data:
            return jsonify({"error": "No content provided"}), 400
        
        content = data.get("content", "")
        if not content or len(content.strip()) < 50:
            return jsonify({"error": "Content too short or empty"}), 400
        
        print(f"Processing webpage content ({len(content)} chars)...")
        
        # Parse HTML and extract text
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())  # Clean up whitespace
        
        if len(text.strip()) < 50:
            return jsonify({"error": "Extracted text too short. Page might be JavaScript-heavy."}), 400
        
        print(f"Extracted {len(text)} characters of text")
        
        # Split and create vector store
        texts = text_splitter.split_text(text)
        print(f"Split into {len(texts)} chunks")
        
        if not texts:
            return jsonify({"error": "No text chunks created"}), 400
        
        vector_store = FAISS.from_texts(texts, embeddings)
        print("✓ Vector store created successfully")
        
        return jsonify({
            "status": "ready",
            "chunks": len(texts),
            "characters": len(text)
        })
    except Exception as e:
        print(f"Error processing webpage: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    """Process YouTube video transcript"""
    global vector_store
    try:
        data = request.json or {}
        video_id = data.get("videoId", "")
        
        if not video_id:
            return jsonify({"error": "No YouTube Video ID provided"}), 400

        print(f"Fetching transcript for video ID: {video_id}")
        
        try:
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find English transcript first
            transcript = None
            try:
                transcript = transcript_list.find_transcript(['en'])
                print("Found English transcript")
            except NoTranscriptFound:
                # Get first available transcript
                try:
                    transcript = next(iter(transcript_list))
                    print(f"Using transcript in language: {transcript.language_code}")
                except StopIteration:
                    return jsonify({
                        "error": "No transcripts available for this video"
                    }), 400
            
            # Fetch transcript data
            transcript_data = transcript.fetch()
            raw_text = " ".join([t['text'] for t in transcript_data])
            
            if not raw_text or len(raw_text.strip()) < 50:
                return jsonify({"error": "Transcript is too short or empty"}), 400
            
            print(f"Fetched {len(raw_text)} characters of transcript")
            
            # Create vector store
            texts = text_splitter.split_text(raw_text)
            print(f"Split into {len(texts)} chunks")
            
            vector_store = FAISS.from_texts(texts, embeddings)
            print("✓ Vector store created successfully")
            
            return jsonify({
                "status": "ready", 
                "message": "YouTube transcript processed.",
                "chunks": len(texts)
            })

        except TranscriptsDisabled:
            return jsonify({
                "error": "Subtitles are disabled for this video. Try a different video."
            }), 400
        except NoTranscriptFound:
            return jsonify({
                "error": "No subtitles found for this video. The creator may not have added subtitles."
            }), 400
        except Exception as transcript_error:
            print(f"Transcript Error: {transcript_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": f"Failed to fetch transcript: {str(transcript_error)}"
            }), 400

    except Exception as e:
        print(f"General YouTube processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    """Process PDF from URL"""
    global vector_store
    try:
        data = request.json
        if not data or "pdf_url" not in data:
            return jsonify({"error": "No PDF URL provided"}), 400
        
        pdf_url = data.get("pdf_url")
        print(f"Fetching PDF from: {pdf_url}")
        
        # Fetch PDF with timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(pdf_url, timeout=30, headers=headers)
        resp.raise_for_status()
        
        # Read PDF
        reader = PdfReader(io.BytesIO(resp.content))
        pdf_text = ""
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            return jsonify({"error": "PDF is empty or could not extract text"}), 400
        
        print(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Split and create vector store
        texts = text_splitter.split_text(pdf_text)
        print(f"Split into {len(texts)} chunks")
        
        vector_store = FAISS.from_texts(texts, embeddings)
        print("✓ Vector store created successfully")
        
        return jsonify({
            "status": "ready",
            "chunks": len(texts),
            "pages": len(reader.pages)
        })
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDF: {e}")
        return jsonify({"error": f"Failed to fetch PDF: {str(e)}"}), 500
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer questions using the vector store"""
    global vector_store
    
    if not vector_store:
        return jsonify({"error": "No content processed yet. Please process a page/video/PDF first."}), 400
    
    try:
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is empty"}), 400
        
        print(f"Question: {question}")
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        
        if not docs:
            return jsonify({"error": "No relevant content found"}), 400
        
        # Build context from documents
        context = "\n\n".join([d.page_content for d in docs])
        
        # Create prompt
        prompt = SYSTEM_TEMPLATE.replace("{context}", context) + f"\n\nUser Question: {question}"
        
        # Get answer
        response = llm.invoke(prompt)
        answer = response.content
        
        print(f"Answer: {answer[:100]}...")
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Error answering question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to answer: {str(e)}"}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Flask Server")
    print("="*50)
    print(f"API Key Set: {bool(os.environ.get('GOOGLE_API_KEY'))}")
    print(f"Models Initialized: {llm is not None and embeddings is not None}")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)