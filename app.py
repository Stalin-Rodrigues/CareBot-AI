from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from google.api_core.exceptions import ResourceExhausted
import os
import time
from functools import lru_cache

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configuration
MAX_TOKENS = 1024  # Increased for flash model
REQUEST_DELAY = 1.0  # Reduced delay since flash is faster
CACHE_SIZE = 200  # Increased cache size

# Get API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    print("ERROR: Missing required API keys in .env file")
    exit(1)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize components
embeddings = download_hugging_face_embeddings()
index_name = "carebot"

try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
except Exception as e:
    print(f"Pinecone initialization error: {str(e)}")
    exit(1)

# Initialize Gemini Flash model with optimized settings
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",  # Using the flash preview model
        temperature=0.2,  # Lower temperature for more focused responses
        max_output_tokens=MAX_TOKENS,
        top_p=0.95,  # Added for better response quality
        top_k=40,  # Added for better response quality
        max_retries=5,  # Increased retries
        request_timeout=60  # Longer timeout for preview model
    )
except Exception as e:
    print(f"Gemini initialization error: {str(e)}")
    # List available models if initialization fails
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Available models:", [m.name for m in genai.list_models()])
    except Exception as e:
        print(f"Failed to list models: {str(e)}")
    exit(1)

# Create chains
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Request throttling and caching
last_request_time = 0

@lru_cache(maxsize=CACHE_SIZE)
def cached_rag_invoke(query):
    global last_request_time
    now = time.time()
    
    if now - last_request_time < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - (now - last_request_time))
    
    last_request_time = time.time()
    return rag_chain.invoke({"input": query})

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a valid question"
    
    try:
        response = cached_rag_invoke(msg)
        answer = response["answer"]
        print("Response:", answer)
        return str(answer)
    except ResourceExhausted:
        return "Our systems are busy. Please wait a few moments and try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)