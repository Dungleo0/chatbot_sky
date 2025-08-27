from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List,Annotated, TypedDict
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEndpointEmbeddings



load_dotenv()

# SQLite memory
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# LLM
GROQ_API_KEY = os.getenv("CHAT_INTENT")


llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=None,
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "chat-index")

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1024,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",  # hoặc BAAI/bge-m3
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

vectorstore = PineconeVectorStore(index, embeddings, "text")



# State
class BasicChatState(dict):
    messages: Annotated[list, add_messages]

def chatbot(state):
    """Node xử lý trả lời từ LLM"""
    user_query = state["messages"][-1].content

    return {
        "messages": [HumanMessage(content=user_query)]
    }

# Graph
graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")
app_graph = graph.compile(checkpointer=memory)

# FastAPI
app = FastAPI()

config = {"configurable": {"thread_id": 1}}

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    """Chat có kết hợp tìm kiếm Pinecone + dữ liệu CSV"""

    # 1. Tìm kiếm trong Pinecone
    docs = vectorstore.similarity_search(request.user_input, k=3)

    context = ""
    if docs:
        context_parts = []
        for d in docs:
            # Chỉ lấy page_content + một vài metadata quan trọng
            meta_preview = {k: str(v)[:50] for k, v in d.metadata.items()}  
            context_parts.append(f"Nội dung: {d.page_content}\nMetadata: {meta_preview}")
        context = "\n\n".join(context_parts)

        # Giới hạn context (để tránh vượt token limit)
        max_chars = 3000
        if len(context) > max_chars:
            context = context[:max_chars] + "... [cắt bớt]"
        
        query_with_context = f"""
Người dùng hỏi: {request.user_input}

Dữ liệu liên quan từ cơ sở tri thức:
{context}

Hãy trả lời ngắn gọn, rõ ràng dựa trên dữ liệu trên.
Nếu dữ liệu chưa đủ, hãy trả lời tự nhiên.
"""
    else:
        # Nếu không tìm thấy dữ liệu → hỏi trực tiếp LLM
        query_with_context = request.user_input

    # 2. Gửi vào LangGraph (chỉ gửi 1 HumanMessage)
    result = app_graph.invoke({
        "messages": [HumanMessage(content=query_with_context)]
    }, config=config)

    return {
        "reply": result["messages"][-1].content,
        "sources": [d.metadata for d in docs] if docs else []
    }



@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload file CSV và lưu dữ liệu vào Pinecone"""
    if not file.filename.endswith(".csv"):
        return {"error": "Chỉ hỗ trợ file CSV"}

    df = pd.read_csv(file.file)

    # Thay thế NaN/null bằng chuỗi rỗng
    df = df.fillna("")

    # Chuyển từng dòng thành text
    texts = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()

    # Tạo id và metadata
    ids = [f"row-{i}" for i in range(len(texts))]

    # Chuyển metadata thành dict và ép kiểu về string/number/boolean
    metadatas = df.to_dict(orient="records")

    # Đảm bảo tất cả metadata đều hợp lệ (convert None → "")
    clean_metadatas = []
    for record in metadatas:
        clean_record = {}
        for k, v in record.items():
            if v is None:
                clean_record[k] = ""  # thay None bằng chuỗi
            elif isinstance(v, (str, int, float, bool)):
                clean_record[k] = v
            else:
                clean_record[k] = str(v)  # ép kiểu về string
        clean_metadatas.append(clean_record)

    # Lưu vào Pinecone
    vectorstore.add_texts(texts, metadatas=clean_metadatas, ids=ids)

    return {"status": "success", "rows_indexed": len(texts)}
