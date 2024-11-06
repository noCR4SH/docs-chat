import gradio as gr
import os

from functions.chat import llm_response
from functions.files import upload_file

if not os.path.exists("data"):
    os.makedirs("data")

with gr.Blocks(title="DocsChat", theme="soft") as app:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🧠📚 DocsChat\nSimply chat wiht your documents.")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("## Chat History")
                            gr.Button("Chat 1")
                            gr.Button("Chat 1")
                            gr.Button("Clean History", variant="primary")
                        with gr.Column(scale=3):
                            gr.ChatInterface(
                                llm_response,
                                type="messages",
                                chatbot=gr.Chatbot(height=800, type="messages")
                            )
                with gr.TabItem("Knowledge"):
                    gr.Markdown("# Knowledge\nUpload and manage your documents.")

                    with gr.Group():
                        file_uploader = gr.File(label="Upload File")
                        upload_button = gr.Button("Upload")

                        upload_button.click(upload_file, file_uploader)

                    with gr.Group():
                        file_explorer = gr.FileExplorer(root_dir="data", label="Uploaded Files")
                        delete_button = gr.Button("Delete Selected")

app.launch()