from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Annotated, TypedDict
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
# Load env
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
# State
class BasicChatState(dict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
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
    result = app_graph.invoke({
        "messages": [HumanMessage(content=request.user_input)]
    }, config=config)

    return {"reply": result["messages"][-1].content}
