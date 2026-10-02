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

persistent_directory= "db/chroma_db"