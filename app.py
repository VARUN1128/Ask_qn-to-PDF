import gradio as gr
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PDFReader
import openai
import os

# Set your API key via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

index = None

def process_pdf(pdf_file):
    global index
    if not pdf_file:
        return "❌ No PDF uploaded."

    try:
        reader = PDFReader()
        documents = reader.load_data(file=pdf_file.name)

        if not documents:
            return "❌ PDF is empty or unreadable."

        index = VectorStoreIndex.from_documents(documents)
        return "✅ PDF uploaded and indexed successfully!"
    
    except Exception as e:
        return f"❌ Failed to process PDF: {str(e)}"

def ask_question(question):
    global index
    if index is None:
        return "❌ Please upload and process a PDF first."

    if not question.strip():
        return "❌ Please enter a question."

    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        return str(response)
    
    except Exception as e:
        return f"❌ Failed to answer question: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Ask Questions to Your PDF (Powered by LlamaIndex & OpenAI)")
    
    with gr.Row():
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("📥 Process PDF")
    
    upload_output = gr.Textbox(label="Upload Status", interactive=False)
    
    question_input = gr.Textbox(label="❓ Your Question")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="💬 Answer", interactive=False)

    upload_btn.click(fn=process_pdf, inputs=pdf_input, outputs=upload_output)
    ask_btn.click(fn=ask_question, inputs=question_input, outputs=answer_output)

demo.launch()
