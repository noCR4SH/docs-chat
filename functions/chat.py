from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
import random
import gradio as gr
import json
from prompts import system_prompts, rag_prompts
from .document_processor import get_relevant_context

llm = ChatOpenAI(model="gpt-4o-mini")

def append_history_to_file(message, role, filename):
    with open(filename, "r") as f:
        history = json.load(f)
    history.append({"role": role, "content": message})

    with open(filename, "w") as f:
        json.dump(history, f)

def llm_response(message, history):
    history_langchain_format = []

    if len(history) == 1:
        if len(message) > 20:
            chat_name = message[:9]
        else:
            chat_name = message
        with open(f"conversation_history/{chat_name}.json", "w") as f:
            json.dump([], f)
    else:
        for msg in history:
            if msg['role'] == "user":
                if len(msg['content']) > 20:
                    chat_name = msg['content'][:9]
                else:
                    chat_name = msg['content']
                break

    # Get relevant context from vector store
    relevant_docs = get_relevant_context(message)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    for msg in history:
        if msg['role'] == 'user':
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            history_langchain_format.append(AIMessage(content=msg['content']))
        elif msg['role'] == "system":
            history_langchain_format.append(SystemMessage(content=msg['content']))

    history_langchain_format.append(HumanMessage(content=rag_prompts.standard_rag_prompt(message, context)))

    llm_response = llm.invoke(history_langchain_format)

    append_history_to_file(message, "user", f"conversation_history/{chat_name}.json")
    append_history_to_file(llm_response.content, "assistant", f"conversation_history/{chat_name}.json")

    return llm_response.content

def new_chat(persona = "default", filename = None):
    print(f"New chat with system prompt: {persona}")

    if persona == "funny":
        system_prompt = [
            {"role": "system", "content": system_prompts.funny}
        ]
    elif persona == "professional":
        system_prompt = [
            {"role": "system", "content": "Only respond in very formal, british language."}
        ]
    else:
        system_prompt = []

    if not filename:
        chatbot = gr.Chatbot(height=800, type="messages", value=system_prompt)
    else:
        with open(f"conversation_history/{filename}", "r") as f:
            history = json.load(f)
        chatbot = gr.Chatbot(height=800, type="messages", value=history)

    return chatbot

def random_response(message, history):
    return random.choice(["Yes", "No"])