import gradio as gr
import os

from functions.chat import llm_response, new_chat
from functions.files import upload_file, delete_files

if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("conversation_history"):
    os.makedirs("conversation_history")

with gr.Blocks(title="DocsChat", theme="soft") as app:
    current_persona = gr.State(None)
    chat_history = gr.State([])
    current_chat = gr.State(None)

    chat_history.value = os.listdir("conversation_history")

    def set_current_persona(value):
        current_persona.value = value
        current_chat.value = None
        chat_history.value = os.listdir("conversation_history")

    def set_current_chat(value = None):
        if value:
            current_chat.value = value
        else:
            current_chat.value = None
        chat_history.value = os.listdir("conversation_history")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🧠📚 DocsChat\nSimply chat wiht your documents.")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("## New Chat")
                            with gr.Group():
                                list_of_personas = gr.Dropdown(label="Select Chat Persona", 
                                                               choices=['default', 'funny', 'professional'],
                                                               type="value",
                                                               value=current_persona.value
                                                               )
                                new_chat_button = gr.Button("New Chat")

                            new_chat_button.click(set_current_persona, list_of_personas)

                            gr.Markdown("## Chat History")
                            with gr.Group():
                                list_of_chats = gr.Dropdown(label="Select Previous Chat", 
                                                            choices=chat_history.value, 
                                                            type="value",
                                                            value=current_chat.value
                                                            )
                                select_chat_button = gr.Button("Load Chat")

                            select_chat_button.click(set_current_chat, list_of_chats)

                            gr.Button("Clean History", variant="primary")
                        with gr.Column(scale=3):

                            @gr.render(triggers=[new_chat_button.click, select_chat_button.click])
                            def chat_interface():
                                gr.ChatInterface(
                                    llm_response,
                                    type="messages",
                                    chatbot=new_chat(current_persona.value, current_chat.value)
                                )
                with gr.TabItem("Knowledge"):
                    gr.Markdown("# Knowledge\nUpload and manage your documents.")

                    with gr.Group():
                        file_uploader = gr.File(label="Upload File", file_types=[".pdf"])
                        upload_button = gr.Button("Upload")

                        upload_button.click(upload_file, file_uploader)

                    with gr.Group():
                        delete_button = gr.Button("Delete Selected", variant="primary", render=False)

                        @gr.render(triggers=[app.load, upload_button.click, delete_button.click])
                        def uploaded_files():
                            file_explorer = gr.FileExplorer(root_dir="data", label="Uploaded Files")

                            delete_button.click(delete_files, file_explorer)
                        delete_button.render()

app.launch()