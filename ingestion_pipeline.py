import os
import sys
import shutil
import time
import re
import uuid
import httpx
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ==========================================
# 1. NETWORKING & SYSTEM CONFIGURATION
# ==========================================
os.environ["HTTPX_ACCEPT_ENCODING"] = "gzip, deflate"
load_dotenv()

DOCS_DIR = "docs"
DB_DIR = "db/chroma_db"

secure_client = httpx.Client(
    headers={"Accept-Encoding": "gzip, deflate"},
    timeout=httpx.Timeout(30.0, connect=10.0)
)

def extract_document_profile(first_page_text: str, filename: str) -> dict:
    """ Identifies document metadata using deterministic parsing patterns """
    search_zone = f"{filename.replace('_', ' ').replace('-', ' ')}\n{first_page_text[:2000]}"
    
    year_match = re.search(r'\b(20\d{2})\b', search_zone)
    fiscal_year = int(year_match.group(1)) if year_match else 2026
    
    quarter = "FY"
    if re.search(r'\b(q1|q-1|first quarter)\b', search_zone, re.IGNORECASE): quarter = "Q1"
    elif re.search(r'\b(q2|q-2|second quarter)\b', search_zone, re.IGNORECASE): quarter = "Q2"
    elif re.search(r'\b(q3|q-3|third quarter)\b', search_zone, re.IGNORECASE): quarter = "Q3"
    elif re.search(r'\b(q4|q-4|fourth quarter)\b', search_zone, re.IGNORECASE): quarter = "Q4"
        
    company_name = filename.split(".")[0].split("_")[0].split("-")[0].strip().title()
    lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
    for line in lines[:5]:
        if any(token in line.lower() for token in ["inc.", "corp", "corporation", "ltd", "limited"]):
            company_name = re.sub(r'(report|transcript|earnings|quarter|financial).*', '', line, flags=re.IGNORECASE).strip()
            break

    ticker = re.sub(r'[^A-Z]', '', company_name.upper())[:4]
    return {
        "company": company_name,
        "ticker": ticker if ticker else "CORP",
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "period": f"{quarter} FY{str(fiscal_year)[-2:]}"
    }

# ==========================================
# 2. ADVANCED PARENT-CHILD STRUCTURAL SPLITTING
# ==========================================
def run_parent_child_ingestion():
    print("🚀 Running Advanced Parent-Child Ingestion Pipeline...")
    
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"❌ Error: The '{DOCS_DIR}' directory is empty.")
        return False

    if os.path.exists(DB_DIR):
        print(f"🗑️ Wiping old database cache folder at '{DB_DIR}'...")
        shutil.rmtree(DB_DIR)
        time.sleep(0.5)

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
    
    # 1. Define Parent Splitter (Captures full comprehensive paragraphs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    # 2. Define Child Splitter (Extracts small granular details for target vector matching)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    final_chunks_to_embed = []

    for file_name in pdf_files:
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"📄 Processing document: {file_name}")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            if not pages: continue
                
            profile = extract_document_profile(pages[0].page_content, file_name)
            
            # First, cut pages into large logical parent sections
            parent_docs = parent_splitter.split_documents(pages)
            
            for p_idx, p_doc in enumerate(parent_docs):
                parent_id = str(uuid.uuid4()) # Generate pointer
                
                # Deduce context type
                p_lower = p_doc.page_content.lower()
                section_type = "Presentation"
                if any(k in p_lower for k in ["q&a", "question-and-answer", "analyst:", "operator:"]):
                    section_type = "Q&A Session"
                
                # Second, split this specific parent document into tiny child snippets
                child_docs = child_splitter.create_documents([p_doc.page_content])
                
                for c_doc in child_docs:
                    # Enriched Child Metadata gets stored in Vector space
                    c_doc.metadata.update({
                        "parent_content": p_doc.page_content, # <-- WE STORE FULL PARENT TEXT IN CHILD METADATA
                        "company": profile["company"],
                        "ticker": profile["ticker"],
                        "period": profile["period"],
                        "quarter": profile["quarter"],
                        "fiscal_year": profile["fiscal_year"],
                        "section": section_type,
                        "source": file_name,
                        "unique_child_id": f"{profile['ticker']}_{p_idx}_{uuid.uuid4().hex[:6]}"
                    })
                    final_chunks_to_embed.append(c_doc)
                    
        except Exception as e:
            print(f"⚠️ Error parsing file {file_name}: {str(e)}")

    if not final_chunks_to_embed:
        print("❌ Ingestion halted. No chunks compiled.")
        return False

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=secure_client)

    print(f"🗄️ Indexing {len(final_chunks_to_embed)} precise child vectors into Chroma DB...")
    Chroma.from_documents(
        documents=final_chunks_to_embed,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("✨ Production Parent-Child Knowledge Base built successfully!")
    return True

if __name__ == "__main__":
    run_parent_child_ingestion()
