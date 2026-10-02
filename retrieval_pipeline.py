import os
import sys
import threading
import time

# ==========================================
# 1. CRITICAL NETWORK BUG INTERCEPTOR
# ==========================================
# Must be declared first to prevent environmental decompressor conflicts
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

# Initialize the modern OpenAI embedding engine
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

# Initialize the Retriever to extract relevant document context chunks
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize the Chat Model with secure network client injection
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    http_client=secure_http_client
)

# Helper function to close down the local runtime execution server port
def shutdown_server():
    print("\n🛑 Shutdown trigger activated. Closing the chat window server interface...")
    # Allows the last message to render in the browser before killing the process thread
    time.sleep(1.2) 
    os._exit(0)


# ==========================================
# 2. CONVERSATIONAL LOGIC ENGINE
# ==========================================
def predict(message, history):
    """
    Processes chat requests, monitors termination commands, 
    and handles modern Gradio history structures safely.
    """
    clean_input = message.strip().lower()
    if clean_input in ["bye", "exit", "quit", "close"]:
        # Launch a background thread to close the app asynchronously
        threading.Timer(0.1, shutdown_server).start()
        return "👋 Operational interface closed. The background server thread has terminated safely. You can now close this browser window."

    # 1. Look up context snippets across the current dataset collection
    relevant_snippets = retriever.invoke(message)
    context_text = "\n\n".join([doc.page_content for doc in relevant_snippets])
    
    # 2. Formulate conversational system rules guiding analytical behaviors
    messages_stack = [
        SystemMessage(
            content=(
                "You are a Senior Equity Analyst. Answer the user's questions in detail "
                "strictly using the provided text context. If the details aren't present in the context, "
                "respond exactly with: 'Based on the information provided, I don't know the answer.'"
            )
        )
    ]
    
    # 3. Dynamic evaluation: Safely checks format before parsing conversation logs
    if history and len(history) > 0:
        first_turn = history[0]
        
        # Format A: Modern Gradio dictionary configuration [{"role": "user", "content": "..."}]
        if isinstance(first_turn, dict):
            for turn in history:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages_stack.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages_stack.append(AIMessage(content=content))
                    
        # Format B: Traditional Gradio nested list configuration [[user_prompt, assistant_response]]
        elif isinstance(first_turn, (list, tuple)) and len(first_turn) == 2:
            for user_prompt, assistant_response in history:
                messages_stack.append(HumanMessage(content=user_prompt))
                messages_stack.append(AIMessage(content=assistant_response))
                
        # Format C: Flat Gradio list configuration fallback
        else:
            for i, content in enumerate(history):
                if i % 2 == 0:
                    messages_stack.append(HumanMessage(content=str(content)))
                else:
                    messages_stack.append(AIMessage(content=str(content)))
        
    # 4. Append current input and retrieved context at the very tail end
    messages_stack.append(
        HumanMessage(content=f"Context from company documents:\n{context_text}\n\nQuestion: {message}")
    )
    
    # 5. Run inference securely over the whole conversation history stack
    execution_result = model.invoke(messages_stack)
    return execution_result.content


# ==========================================
# 3. GRADIO BROWSER INTERFACE LAYOUT
# ==========================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 📈 Mandar's Equity Analyst Agent Workspace")
    gr.Markdown("Ask back-and-forth analytical questions regarding your uploaded corporate files. Type **bye** or **exit** to close the session.")
    
    gr.ChatInterface(
        fn=predict,
        textbox=gr.Textbox(
            placeholder="What are the positives and negatives of the quarterly results?", 
            container=False, 
            scale=7
        ),
        examples=[
            "What are the positives and negatives of the quarterly results?",
            "What are the challenges faced by the company?",
            "exit"
        ],
        cache_examples=False
    )

if __name__ == "__main__":
    print("🚀 Launching localized chat UI hosting node at http://127.0.0.1:7860")
    # Fire up the interface server securely
    demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=False)
