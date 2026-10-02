import os
import sys
import threading
import time

# ==========================================
# 1. CRITICAL NETWORK BUG INTERCEPTOR
# ==========================================
os.environ["HTTPX_ACCEPT_ENCODING"] = "gzip, deflate"

import httpx
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load credentials from your local .env file
load_dotenv()

# Configure an isolated connection pool stripped of broken decompression extensions
secure_http_client = httpx.Client(
    headers={"Accept-Encoding": "gzip, deflate"}
)

persistent_directory = "db/chroma_db"

# Initialize the embedding engine
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    http_client=secure_http_client
)

# Connect to the local Chroma DB instance
db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"} 
)

# Initialize the Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize the Chat Model with network client injection
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    http_client=secure_http_client
)

# Helper function to close down the local runtime execution stack
def shutdown_server():
    print("\n🛑 Shutdown trigger activated. Closing the chat window server interface...")
    # Allows the last message to render in the browser before killing the thread
    time.sleep(1.5) 
    os._exit(0)

# ==========================================
# 2. CHAT LOGIC WITH SHUTDOWN COMMAND HOOK
# ==========================================
def predict(message, history):
    """
    Processes chat requests and monitors inputs for termination commands.
    """
    clean_input = message.strip().lower()
    
    # Check if the user intends to close the session
    if clean_input in ["bye", "exit", "quit", "close"]:
        # Launch a background thread to close the app asynchronously
        threading.Timer(0.1, shutdown_server).start()
        return "👋 Thank you for using the Equity Analyst Agent workspace. The session has ended, and the local server is shutting down now. You can safely close this browser window."

    # 1. Search Chroma DB for relevant PDF text snippets based on the latest question
    relevant_docs = retriever.invoke(message)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Build the System Prompt guiding the analytical behavior
    langchain_messages = [
        SystemMessage(
            content=(
                "Consider yourself a Senior Equity Analyst. Answer the user's question "
                "in comprehensive detail strictly using the provided source text context. "
                "If you cannot find the answer in the context, state: "
                "'Based on the information provided, I don't know the answer.'"
            )
        )
    ]
    
    # 3. Append previous turns from historical state so the LLM remembers past questions
    for user_turn, ai_turn in history:
        langchain_messages.append(HumanMessage(content=user_turn))
        langchain_messages.append(AIMessage(content=ai_turn))
        
    # 4. Inject the latest retrieved context along with the current question at the tail
    current_turn_content = f"Context from company documents:\n{context_text}\n\nQuestion: {message}"
    langchain_messages.append(HumanMessage(content=current_turn_content))
    
    # 5. Run inference securely over the whole conversation history stack
    response = model.invoke(langchain_messages)
    return response.content

# ==========================================
# 3. GRADIO BROWSER INTERFACE LAYOUT
# ==========================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 📈 Equity Analyst Agent Workspace")
    gr.Markdown("Ask back-and-forth analytical questions regarding your uploaded corporate files. Type **bye** or **exit** to close the session.")
    
    gr.ChatInterface(
        fn=predict,
        textbox=gr.Textbox(placeholder="What are the positives and negatives of the quarterly results?", container=False, scale=7),
        examples=[
            "What are the positives and negatives of the quarterly results?",
            "What are the challenges faced by the company?",
            "exit"
        ],
        cache_examples=False
    )

if __name__ == "__main__":
    print("🚀 Initializing localized chat UI layer...")
    # Fire up the interface server
    demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=False)
