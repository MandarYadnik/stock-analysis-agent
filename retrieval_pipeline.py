import os
import sys
import threading
import time
import httpx
import gradio as gr
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ==========================================
# 1. NETWORKING & SYSTEM CONFIGURATION
# ==========================================
os.environ["HTTPX_ACCEPT_ENCODING"] = "gzip, deflate"
load_dotenv()

DB_DIR = "db/chroma_db"
secure_client = httpx.Client(headers={"Accept-Encoding": "gzip, deflate"})

# Dense Retrieval Embedding Model Initialization
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=secure_client)

if not os.path.exists(DB_DIR):
    print(f"❌ Error: Database directory '{DB_DIR}' not found. Run ingestion_pipeline.py first.")
    sys.exit(1)

db = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)

# Retrieve top 5 child matches to gather a wide spread of specific data coordinates
retriever = db.as_retriever(search_kwargs={"k": 5})

# Upgraded Chat Model with expanded tokens to support highly detailed reports
model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0, 
    max_tokens=4000,  # <-- CRITICAL: Prevents the model from cutting off long, descriptive answers
    http_client=secure_client
)

def graceful_shutdown():
    print("\n🛑 Shutdown trigger activated. Closing server workspace...")
    time.sleep(1.0)
    os._exit(0)

# ==========================================
# 2. ADVANCED CONVERSATIONAL RETRIEVAL SWITCH
# ==========================================
def predict(message, history):
    clean_input = message.strip().lower()
    if clean_input in ["bye", "exit", "quit", "close"]:
        threading.Timer(0.1, graceful_shutdown).start()
        return "👋 Workspace offline. You can safely close this browser window."

    # 1. Dense Search Execution (finds precise child sentences rows)
    child_matches = retriever.invoke(message)
    
    parent_context_blocks = []
    metadata_citations = set()
    
    # 2. SWAP LOOP: Pull the large parent content out of the matched child's metadata
    for child in child_matches:
        parent_text = child.metadata.get("parent_content")
        if parent_text and parent_text not in parent_context_blocks:
            parent_context_blocks.append(parent_text)
            
        source = child.metadata.get("source", "PDF")
        period = child.metadata.get("period", "Unknown")
        section = child.metadata.get("section", "General")
        metadata_citations.add(f"• {source} ({period}) — Section: {section}")
        
    full_context_evidence = "\n\n--- Context Section ---\n\n".join(parent_context_blocks)
    citations_text = "\n".join(list(metadata_citations))

    # 3. FIX FOR ISSUE 2: Re-engineered system prompt forcing detailed, informative outputs
    messages_stack = [
        SystemMessage(
            content=(
                "You are an elite Senior Equity Analyst specializing in exhaustive corporate intelligence. "
                "Your objective is to provide comprehensive, deeply informative, detailed, and institutional-grade financial analysis. "
                "Analyze the provided context thoroughly to answer the user's inquiry.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Do not summarize or synthesize high-level overviews. Be as verbose, detailed, and complete as possible.\n"
                "- Present exact financial metrics, percentage movements, revenue values, and numbers exactly as written.\n"
                "- Trace and extract structural causations, management justifications, macro headwinds, and structural risks explicitly.\n"
                "- Use bolding, structured bullet points, and distinct sections to separate different analytical points clearly.\n"
                "- If the provided text context does not contain the answer, respond exactly with: 'Based on the information provided, I don't know the answer.'"
            )
        )
    ]

    # 4. Flexible history parser to digest list-of-dicts or list-of-tuples safely
    if history and len(history) > 0:
        first_turn = history[0]
        
        if isinstance(first_turn, dict):
            for turn in history:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user": messages_stack.append(HumanMessage(content=content))
                elif role == "assistant": messages_stack.append(AIMessage(content=content))
        elif isinstance(first_turn, (list, tuple)) and len(first_turn) == 2:
            for u, a in history:
                messages_stack.append(HumanMessage(content=u))
                messages_stack.append(AIMessage(content=a))
        else:
            for idx, content in enumerate(history):
                if idx % 2 == 0: messages_stack.append(HumanMessage(content=str(content)))
                else: messages_stack.append(AIMessage(content=str(content)))

    # 5. Inject full parent context alongside the query
    messages_stack.append(
        HumanMessage(content=f"Factual Evidentiary Context:\n{full_context_evidence}\n\nQuestion: {message}")
    )

    # 6. Secure inference run
    response = model.invoke(messages_stack)
    return f"{response.content}\n\n📊 **Verified Analytical Sources:**\n{citations_text}"

# ==========================================
# 3. FIX FOR ISSUE 1: FULL SCREEN LAYOUT
# ==========================================
# Passing fill_height=True instructs the layout blocks to take 100% of browser window space
with gr.Blocks(theme="soft", fill_height=True) as demo:
    gr.Markdown("# 📈 Mandar's Equity Analyst Agent Workspace")
    gr.Markdown("Analyzing corporate data streams via dynamic Parent-Child semantic recovery loops.")
    
    gr.ChatInterface(
        fn=predict, 
        fill_height=True,  # <-- Makes the chat UI stretch completely to the bottom of the screen
        textbox=gr.Textbox(
            placeholder="Ask for precise financial metrics, trend reviews, or cross-quarter comparisons...", 
            scale=7
        ), 
        cache_examples=False
    )

if __name__ == "__main__":
    print("🚀 Launching full-screen chat UI hosting node at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=False)
