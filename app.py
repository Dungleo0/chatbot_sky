from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List,Annotated, TypedDict
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import random


load_dotenv()

# SQLite memory
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# LLM
GROQ_API_KEY = os.getenv("CHAT_INTENT")


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
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
    model="BAAI/bge-m3", 
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

vectorstore = PineconeVectorStore(index, embeddings, "text")



# State
class BasicChatState(dict):
    messages: Annotated[list, add_messages]

def chatbot(state):
    """Node xử lý trả lời từ LLM"""
    user_query = state["messages"][-1].content

    result = llm.invoke([HumanMessage(content=user_query)])

    if isinstance(result, str):
        result = AIMessage(content=result)

    return {"messages": [result]}

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
    docs = vectorstore.similarity_search(request.user_input, k=7)

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
Nếu người dùng hỏi không liên quan đến cơ sở tri thức, hãy nói là chưa hỗ trợ chỉ có thể tư vấn về vấn đề trong cơ sở dữ liệu
Người dùng hỏi: {request.user_input}

Dữ liệu liên quan từ cơ sở tri thức:
{context}

Hãy trả lời ngắn gọn, rõ ràng dựa trên dữ liệu trên.
"""
    else:
        # Nếu không tìm thấy dữ liệu → hỏi trực tiếp LLM
        query_with_context = request.user_input

    result = app_graph.invoke({
        "messages": [HumanMessage(content=query_with_context)]
    }, config=config)

    return {
        "reply": result["messages"][-1].content,
        "sources": [d.metadata for d in docs] if docs else []
    }

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload file CSV, chuẩn hóa dữ liệu, chunk và lưu vào Pinecone"""

    # Kiểm tra định dạng
    if not file.filename.endswith(".csv"):
        return {"error": "Chỉ hỗ trợ file CSV"}

    # Đọc CSV an toàn
    try:
        df = pd.read_csv(file.file).fillna("")
    except Exception as e:
        return {"error": f"Lỗi đọc CSV: {str(e)}"}

    # Nếu có nhiều dòng trùng id_product -> lấy bản ghi đầu tiên
    if "id_product" in df.columns:
        df = df.groupby("id_product", as_index=False).first()

    # Text splitter để chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", ";", " "]
    )

    texts, metadatas, ids = [], [], []

    for idx, row in df.iterrows():
        # Bảo đảm có product_id
        product_id = str(row.get("id_product", "")).strip()
        if not product_id:
            product_id = f"prod_{uuid.uuid4().hex[:8]}"

        # Nội dung để embed
        text_to_embed = (
            f"Tên sản phẩm: {str(row.get('name', '')).strip()}\n"
            f"Hãng: {str(row.get('brand', '')).strip()}\n"
            f"Giá gốc: {str(row.get('price', '')).strip()}\n"
            f"Khuyến mãi: {str(row.get('coupon', '')).strip()}\n"
            f"Giá chưa giảm: {str(row.get('price_old', '')).strip()}\n"
            f"Mô tả chi tiết: {str(row.get('detail_description', '')).strip()}"
        )

        # Chunk mô tả
        chunks = text_splitter.split_text(text_to_embed)

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            ids.append(f"{product_id}_{i}")  # ID duy nhất

            meta = {
                "id_product": product_id,
                "chunk_id": str(i),
                "name": str(row.get("name", "")).strip(),
                "brand": str(row.get("brand", "")).strip(),
                "price": str(row.get("price", "")).strip(),
                "coupon": str(row.get("coupon", "")).strip(),
                "price_old": str(row.get("price_old", "")).strip(),
            }
            metadatas.append(meta)

    # Lưu vào Pinecone bằng upsert (tránh lỗi trùng ID)
    try:
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        return {"error": f"Lỗi lưu vào Pinecone: {str(e)}"}

    return {
        "status": "success",
        "products_indexed": len(df),
        "chunks_indexed": len(texts)
    }



@app.get("/")
def root():
    return {
        "message": "truy cap /docs"
    }

