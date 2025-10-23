import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dotenv import load_dotenv
load_dotenv() # This loads the GOOGLE_API_KEY from your .env file

import io 
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- THIS IS THE CORRECTED IMPORT BLOCK ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
# --- END OF CORRECTED IMPORTS ---


# --- 1. Configuration ---
# (Your .env file handles the key)

# --- 2. App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- 3. Global Variables ---
vector_store = None
llm = None
embeddings = None
text_splitter = None
youtube_api_client = YouTubeTranscriptApi()

# --- 4. Custom Prompt Template ---
SYSTEM_TEMPLATE = """
You are a helpful assistant. Your primary goal is to answer questions about the webpage, PDF, or video transcript context provided.
First, try to answer the user's question based *only* on the context document provided.
If the information *is* in the context, provide a detailed answer based on it.
If the information is *not* in the context, politely tell the user you couldn't find it on the page, and then try to answer their question as a general AI assistant.
Always format your answer to be clear and easy to read. Use bullet points or short paragraphs if appropriate.

Here is the context:
{context}
"""

# --- 5. Initialize LLM & Embeddings ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3) 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
except Exception as e:
    print(f"Error initializing Google AI models: {e}")


# --- 6. Helper Function for Translation ---
def get_english_transcript(transcript_text):
    if not llm: return transcript_text 
    try:
        prompt = f"Detect the language of the following text. Respond with only the two-letter ISO 639-1 language code... Text: '{transcript_text[:500]}'"
        lang_code = llm.invoke(prompt).content.strip().lower()
        if 'en' in lang_code:
            return transcript_text
        else:
            translation_prompt = f"Translate the following text to English: '{transcript_text}'"
            return llm.invoke(translation_prompt).content
    except Exception as e:
        print(f"Error during translation: {e}")
        return transcript_text


# --- 7. "Process Webpage" Endpoint ---
@app.route('/process_webpage', methods=['POST'])
def process_webpage():
    global vector_store 
    if not llm: return jsonify({"error": "Models not initialized."}), 500
    try:
        data = request.json
        page_content = data.get('content', '')
        if not page_content: return jsonify({"error": "No content provided"}), 400
        soup = BeautifulSoup(page_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        docs = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(docs, embeddings) 
        print("Webpage processed successfully!")
        return jsonify({"status": "ready", "message": f"Processed {len(text)} characters."})
    except Exception as e:
        print(f"Error processing page: {e}")
        return jsonify({"error": str(e)}), 500


# --- 8. "Process YouTube" Endpoint ---
@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    global vector_store, youtube_api_client
    if not llm: return jsonify({"error": "Models not initialized."}), 500
    try:
        data = request.json
        video_id = data.get('videoId', '')
        if not video_id: return jsonify({"error": "No YouTube Video ID provided"}), 400
        print(f"Fetching transcript for video ID: {video_id}")
        try:
            transcript_list = youtube_api_client.list(video_id)
            transcript = transcript_list.find_transcript(['en', 'hi', 'es', 'de', 'fr', 'ja', 'ko', 'ru'])
            transcript_data = transcript.fetch()
            raw_transcript = " ".join([item.text for item in transcript_data])
            english_transcript = get_english_transcript(raw_transcript)
        except (NoTranscriptFound, TranscriptsDisabled):
            return jsonify({"error": "No transcript available for this video."}), 400
        
        docs = text_splitter.split_text(english_transcript)
        vector_store = FAISS.from_texts(docs, embeddings) 
        print("YouTube transcript processed successfully!")
        return jsonify({"status": "ready", "message": f"Processed {len(english_transcript)} characters."})
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
        return jsonify({"error": str(e)}), 500

# --- 9. "Process PDF" Endpoint ---
@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global vector_store
    if not llm: return jsonify({"error": "Models not initialized."}), 500

    try:
        data = request.json
        pdf_url = data.get('pdf_url', '')
        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400

        print(f"Fetching PDF from: {pdf_url}")
        
        response = requests.get(pdf_url)
        response.raise_for_status() 

        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
        
        if not pdf_text:
            return jsonify({"error": "Could not extract text from this PDF."}), 400

        docs = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(docs, embeddings) 
        
        print("PDF processed successfully!")
        return jsonify({"status": "ready", "message": f"Processed {len(pdf_text)} characters from PDF."})

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({"error": f"Failed to process PDF. Is the URL correct and public? {e}"}), 500


# --- 10. "Ask Question" Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store, llm
    if not vector_store: return jsonify({"error": "Page not processed yet."}), 400
    try:
        data = request.json
        question = data.get('question', '')
        history_list = data.get('history', [])
        if not question: return jsonify({"error": "No question provided"}), 400
        chat_history = []
        for item in history_list:
            if item.get('type') == 'human': chat_history.append(HumanMessage(content=item.get('content')))
            elif item.get('type') == 'ai': chat_history.append(AIMessage(content=item.get('content')))
        qa_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_TEMPLATE), ("human", "{question}")])
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), combine_docs_chain_kwargs={"prompt": qa_prompt})
        response = qa_chain.invoke({"question": question, "chat_history": chat_history})
        print(f"Q: {question}\nA: {response.get('answer')}")
        return jsonify({"answer": response.get('answer')})
    except Exception as e:
        print(f"Error asking question: {e}")
        return jsonify({"error": str(e)}), 500

# --- 11. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)