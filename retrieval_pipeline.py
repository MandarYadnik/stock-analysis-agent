import os
import sys

# ==========================================
# 1. CRITICAL NETWORK BUG INTERCEPTOR
# ==========================================
os.environ["HTTPX_ACCEPT_ENCODING"] = "gzip, deflate"

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Load credentials from your local .env file
load_dotenv()

# Configure an isolated connection pool stripped of broken decompression extensions
secure_http_client = httpx.Client(
    headers={"Accept-Encoding": "gzip, deflate"}
)

persistent_directory = "db/chroma_db"

# Initialize the modern OpenAI embedding engine with the clean client
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    http_client=secure_http_client  # Enforces clean network connections
)

# Connect to the local Chroma DB instance
db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"} 
)

# Define query
#query = "what are the challenges faced by the company?"
query = "What are the positives and negatives of the quarterly results ?"

# Retrieve relevant snippets from the PDF files
retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print(f"Retrieved {len(relevant_docs)} context chunks from Chroma DB.")

# --- Combine the text chunks into a unified string context ---
context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

# ==========================================
# 2. CHAT MODEL WITH NETWORK INJECTION
# ==========================================
# FIX: Injecting the clean client to bypass the httpx2 bug during text generation
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    http_client=secure_http_client  # <--- THIS STOPS THE COMPRESSION CRASH HERE
)

# Constructing messages with factual PDF context injected
messages = [
    SystemMessage(
        content=(
            "Consider yourself a Senior Equity Analyst. Answer the user's question "
            "Generate the answer in detail"
            "strictly using the provided source text context. If you cannot find the "
            "answer in the context, state: 'Based on the information provided, I don't know the answer.'"
        )
    ),
    HumanMessage(
        content=f"Context from company documents:\n{context_text}\n\nQuestion: {query}"
    )
]

# Run the inference engine securely
result = model.invoke(messages)

print("\n📊 Final Analyst Output:")
print(result.content)
