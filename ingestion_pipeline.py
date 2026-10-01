import os
import sys

# ==========================================
# 1. CRITICAL NETWORK BUG INTERCEPTOR
# ==========================================
# This MUST run before any LangChain or OpenAI dependencies are loaded to stop the httpx2 bug
os.environ["HTTPX_ACCEPT_ENCODING"] = "gzip, deflate"

import httpx
from dotenv import load_dotenv
# Upgraded structural imports matching the latest LangChain ecosystem
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load credentials from your local .env file
load_dotenv()

# ==========================================
# 2. ROBUST DOCUMENT INGESTION
# ==========================================
def load_documents(docs_path="docs"):
    """
    Scans the targeted folder and extracts text pages using PyPDFLoader.
    Replaces DirectoryLoader to isolate environment library dependencies.
    """
    print(f"🔍 Scanning directory: '{docs_path}'...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"❌ Error: The directory '{docs_path}' does not exist.")
    
    # Read files explicitly to prevent silent directory-skipping anomalies
    pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        raise FileNotFoundError(f"❌ Error: No valid PDF files found inside the '{docs_path}' directory.")
        
    all_loaded_pages = []
    
    for file_name in pdf_files:
        full_file_path = os.path.join(docs_path, file_name)
        print(f"📄 Loading document chunks from: {file_name}")
        try:
            file_loader = PyPDFLoader(full_file_path)
            pages = file_loader.load()
            all_loaded_pages.extend(pages)
        except Exception as file_error:
            print(f"⚠️ Failed parsing target file {file_name}: {str(file_error)}")
            
    print(f"📊 Extraction Complete: Fetched {len(all_loaded_pages)} data pages.")
    return all_loaded_pages


# ==========================================
# 3. SEMANTIC TEXT SEGMENTATION
# ==========================================
def split_documents(documents, chunk_size=1500, chunk_overlap=200):
    """
    Splits text data recursively by paragraphs and sentences.
    1500 character limits with 200 character overlap maximize 
    retrieval density for financial financial models.
    """
    print("✂️ Splitting extracted text into semantic chunks...")
    
    # Recursive splitting handles mathematical text layout maps perfectly
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Generated {len(chunks)} contextual snippets.")
    return chunks


# ==========================================
# 4. CHROMA VECTOR VECTOR STORE PERSISTENCE
# ==========================================
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """
    Creates and stores numeric vector matrices into a local Chroma DB deployment.
    Forces standard HTTPX clients to bypass package manipulation errors.
    """
    print("------- Initializing Vector Store Construction -------")
    
    # Configure an isolated connection pool stripped of broken decompression extensions
    secure_http_client = httpx.Client(
        headers={"Accept-Encoding": "gzip, deflate"}
    )
    
    # Initialize the modern OpenAI embedding engine
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        http_client=secure_http_client  # Enforces clean network connections
    )
    
    print(f"🗄️ Writing data tables directly onto disk at: '{persist_directory}'...")
    
    # Generate database entities and compile vector charts
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} # Optimizes for similarity angle math
    )
    
    print("------- Completed Creating Vector Store Successfully -------")
    return vectorstore


# ==========================================
# 5. ORCHESTRATION ENGINE
# ==========================================
def main():
    print("🚀 Initializing Ingestion Pipeline Execution Flow...")
    
    # Runtime assertion: Stop early if api token bindings are broken
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY is not defined in your environment hooks.")
        print("Please ensure your .env file is situated in this working directory.")
        sys.exit(1)
        
    try:
        # Step 1: Read files from disk
        raw_documents = load_documents(docs_path="docs")
        
        # Step 2: Segment raw files into chunks
        processed_chunks = split_documents(raw_documents)
        
        # Step 3: Embed vectors and save to database
        create_vector_store(processed_chunks)
        
        print("\n🎉 Success: System assets processed and database created safely!")
        
    except Exception as pipeline_failure:
        print(f"\n❌ Pipeline execution terminated unexpectedly: {str(pipeline_failure)}")


if __name__ == "__main__":
    main()
