import os
import shutil
import gradio as gr

UPLOAD_FOLDER = "data"

def upload_file(file):
    shutil.copy(file, UPLOAD_FOLDER)
    gr.Info("File Uploaded!")