import os
import requests # Import requests for making HTTP calls to Gemini API
import shutil # For removing directories
import time # Import time for sleep
import io # Add this import
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from peft import PeftModel # No longer needed if loading already merged model
import torch

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
# Changed BASE_MODEL_PATH to point to the already merged final model
BASE_MODEL_PATH = "models/medical_llm_merged_final" 
# ADAPTER_PATH is no longer directly used for loading in app.py if BASE_MODEL_PATH is the merged model
ADAPTER_PATH = "models/adapter_after_extra_training" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Directory for model offloading (REQUIRED by accelerate if model is too large for RAM/VRAM)
OFFLOAD_DIR = "model_offload_cache" 
os.makedirs(OFFLOAD_DIR, exist_ok=True) # Ensure the directory exists

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# === Flask Setup ===
app = Flask(
    __name__,
    static_folder='frontend/dist/assets',
    template_folder='frontend/dist'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disable caching for development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# === Global Variables ===
user_vector_stores = {}

# === Local Model Setup ===
print(f"Loading final merged model from: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# Load the already merged model directly
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map="auto",
    offload_folder=OFFLOAD_DIR # Still crucial for large models
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print(f"Creating text generation pipeline on device: {DEVICE}")
gen_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    # Removed the 'device' argument because device_map="auto" already handles placement
    # device=0 if DEVICE == "cuda" else -1, 
    max_new_tokens=128, # Increased to 128 for slightly longer raw answers
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# === Embeddings ===
print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Helper: Allowed File Type ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Summarizer (Optional: for a separate summary feature, not for RAG context) ===
def summarize_text(text, num_sentences=5):
    """Summarizes text using TextRank algorithm."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Could not generate summary."

# === Extract ALL Text from PDF for RAG ===
def extract_full_text_from_pdf(pdf_path):
    """
    Extracts all text content from a PDF by first reading it into memory.
    Returns an empty string if extraction fails, to prevent error messages
    from polluting the LLM context.
    """
    full_text = []
    try:
        # Read the file content into a BytesIO object
        with open(pdf_path, 'rb') as f:
            pdf_bytes = io.BytesIO(f.read())
        
        # Pass the in-memory bytes to PdfReader
        reader = PdfReader(pdf_bytes)
        
        # Check if PDF is encrypted and PyCryptodome is missing
        if reader.is_encrypted:
            try:
                # Attempt to decrypt (requires PyCryptodome)
                reader.decrypt('') # Try with empty password first
            except utils.PdfReadError as e:
                print(f"PDF encryption error (needs password or PyCryptodome): {e}")
                # Return empty string if decryption fails due to missing PyCryptodome or wrong password
                return "" # Return empty string, not an error message
            except Exception as e:
                print(f"General PDF decryption error: {e}")
                return "" # Return empty string on other decryption failures

        for page in reader.pages:
            content = page.extract_text()
            if content:
                full_text.append(content)
        return "\n".join(full_text).strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        # Do NOT return the error message as text. Return empty string.
        return ""

# === Vector Store Creation ===
def create_vector_store(text):
    """Creates a FAISS vector store from text chunks."""
    if not text:
        raise ValueError("Cannot create vector store from empty text.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding_model)

# === Prompt Template for RAG ===
rag_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful medical assistant. Use the following context to answer the question.\n"
        "If the answer is not in the context, state that you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Use the following **bold section headings** if relevant to structure your answer:\n"
        "**Question Summary**, **Key Findings**, **Test Results**, "
        "**Trends or Anomalies**, **Warnings**, **Suggestions**, **Explanation of Terms**\n\n"
        "Question: {question}\n\nAnswer:"
    )
)

# === Gemini Formatting Function ===
def get_formatted_gemini_response(raw_medical_text):
    """
    Sends the raw medical text to Gemini for formatting and summarization,
    with a specific focus on structured points and desired length.
    """
    prompt = f"""The following is a medical analysis or answer generated from a document.
    Please rephrase and summarize it concisely for a healthcare professional.
    Provide a detailed answer to the question, including relevant test results,
    standard limits, and clear recommendations. Aim for approximately 10-12 lines of total output.

    Structure the response using these exact bold headings if applicable, followed by bullet points for details:

    **Key Findings:**
    - ...
    **Test Results:**
    - ...
    **Normal Range:**
    - ...
    **Interpretation:**
    - ...
    **Recommendations:**
    - ...
    **Explanation of Terms:**
    - ...

    If a section is not relevant or has no information, omit that heading.
    Keep sentences concise but informative.

    Medical Content:
    {raw_medical_text}
    """

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2, # Keep temperature low for factual rephrasing
            "maxOutputTokens": 400 # Adjusted to encourage 10-12 lines
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result and result.get('candidates'):
            first_candidate = result['candidates'][0]
            if first_candidate.get('content') and first_candidate['content'].get('parts'):
                return first_candidate['content']['parts'][0]['text']
        return "Gemini could not generate a formatted response."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error connecting to formatting service: {e}"
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return f"Error processing formatting response: {e}"


# === Routes ===
@app.route("/")
def serve_react_app():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/<path:filename>')
def serve_root_static_files(filename):
    return send_from_directory(app.template_folder, filename)

@app.route("/upload", methods=["POST"])
def upload():
    user_id = session.get('user_id', request.remote_addr)
    session['user_id'] = user_id

    file = request.files.get("file")
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid PDF file."}), 400


    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    try:
        file.save(filepath)
        print(f"File saved to: {filepath}")

        extracted_text = extract_full_text_from_pdf(filepath)
        if not extracted_text.strip():
            # If text extraction fails, provide a specific error message
            return jsonify({"error": "Failed to extract text from PDF. It might be encrypted, image-based, or corrupted."}), 400

        vector_store = create_vector_store(extracted_text)
        retriever = vector_store.as_retriever(search_type="similarity", k=3)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": rag_prompt_template},
            return_source_documents=False
        )
        user_vector_stores[user_id] = qa_chain

        return jsonify({"message": "PDF uploaded and processed successfully! You can now ask questions."})

    except Exception as e:
        print(f"[ERROR] /upload route: {e}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}. Please try again or check the PDF content."}), 500
    finally:
        # Ensure the file is closed and then removed
        if os.path.exists(filepath):
            # Retry deletion a few times with a small delay
            for i in range(3): # Try up to 3 times
                try:
                    os.remove(filepath)
                    print(f"Cleaned up uploaded file: {filepath}")
                    break # Exit loop if successful
                except OSError as e:
                    print(f"Attempt {i+1} to remove file {filepath} failed: {e}. Retrying in 0.5 seconds...")
                    time.sleep(0.5) # Wait a bit before retrying
            else: # This block executes if the loop completes without a 'break'
                print(f"Failed to remove file {filepath} after multiple attempts.")


@app.route("/ask", methods=["POST"])
def ask():
    user_id = session.get('user_id')
    if not user_id or user_id not in user_vector_stores:
        return jsonify({"answer": "**Error:** Please upload a medical report first."})

    qa_chain = user_vector_stores[user_id]

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "**Error:** Please enter a valid question."})

        print(f"[INFO] User '{user_id}' received question: '{question}'")

        # 1. Get initial answer from local LLM (fine-tuned)
        local_llm_result = qa_chain.invoke({"query": question})
        raw_answer = local_llm_result.get('result', str(local_llm_result)) if isinstance(local_llm_result, dict) and 'result' in local_llm_result else str(local_llm_result)
        print(f"[INFO] Raw answer from local LLM: {raw_answer[:200]}...")

        # 2. Pass raw answer to Gemini for formatting and summarization
        formatted_answer = get_formatted_gemini_response(raw_answer)
        print(f"[INFO] Formatted answer from Gemini: {formatted_answer[:200]}...")

        return jsonify({"answer": formatted_answer})

    except Exception as e:
        print(f"[ERROR] /ask route: {e}")
        return jsonify({"answer": "**Error:** An internal server error occurred while processing your question."})

# === Run App ===
if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        # Clean up the offload directory on app shutdown
        if os.path.exists(OFFLOAD_DIR):
            print(f"Cleaning up offload directory: {OFFLOAD_DIR}")
            shutil.rmtree(OFFLOAD_DIR)

