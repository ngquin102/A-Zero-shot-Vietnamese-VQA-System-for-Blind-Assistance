import gradio as gr
import os
import numpy as np
import time
import threading
from queue import Queue
import sounddevice as sd
from PIL import Image
from io import BytesIO
from src.utils import get_device
from src.image_processor import load_image
from src.model_loader import ImageQAModel
from app import ImageQASystem
from src.speech_utils import SpeechProcessor 


GEMINI_API_KEY = "..............."  
qa_system = ImageQASystem(gemini_api_key=GEMINI_API_KEY)

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ Hệ Thống Hỏi Đáp Hình Ảnh Hỗ Trợ Người Khiếm Thị với Zero-Shot Learning")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(label="📂 Tải ảnh từ máy tính", type="filepath")
                caption_output = gr.Textbox(label="📜 Thông tin trích xuất từ ảnh", lines=4)
                image_display = gr.Image(label="🖼️ Ảnh đã tải lên", type="pil")

                with gr.Row():
                    upload_btn = gr.Button("📸 Xử Lý Ảnh")
                    clear_btn = gr.Button("🗑️ Xóa")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("💬 Hỏi đáp văn bản"):
                        chatbot = gr.Chatbot(label="💬 Hội Thoại Về Ảnh", height=300)
                        msg = gr.Textbox(label="📝 Nhập câu hỏi về ảnh")

                        with gr.Row():
                            submit_btn = gr.Button("🚀 Gửi")
                            clear_chat_btn = gr.Button("🗑️ Xóa Hội Thoại")
                    
                    with gr.TabItem("🎤 Hỏi đáp giọng nói"):
                        with gr.Row():
                            record_btn = gr.Button("🎤 Bắt đầu ghi âm (5 giây)")
                            
                        voice_result = gr.Textbox(label="💬 Kết quả hỏi đáp bằng giọng nói", lines=5)
                        question_text = gr.Textbox(label="🔍 Câu hỏi đã nhận dạng", visible=True)
                        audio_output = gr.Audio(label="🔊 Phản hồi bằng giọng nói", format="mp3")

                        gr.Markdown("""
                        ### 🎙️ Hoặc ghi âm câu hỏi:
                        """)

                        audio_input = gr.Audio(type="numpy", label="🎤 Ghi âm câu hỏi trực tiếp")  

        upload_btn.click(
            fn=lambda path: qa_system.process_image(path),
            inputs=[image_input],
            outputs=[caption_output, image_display]
        )

        def handle_question(question, chat_history):
            answer = qa_system.answer_question(question)
            return "", chat_history + [[question, answer]]

        submit_btn.click(handle_question, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(handle_question, inputs=[msg, chatbot], outputs=[msg, chatbot])
        

        record_btn.click(
            fn=qa_system.record_and_transcribe,
            inputs=[],
            outputs=[voice_result, question_text, audio_output]
        )
        
        audio_input.change(
            fn=qa_system.process_audio_input,
            inputs=[audio_input],
            outputs=[voice_result, question_text, audio_output]
        )

        clear_btn.click(lambda: ("", None), outputs=[caption_output, image_display])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])

        gr.Markdown("""
        ## 📖 Hướng Dẫn Sử Dụng
        1️⃣ **Chọn ảnh từ máy tính**  
        2️⃣ **Nhấn "Xử Lý Ảnh"** để trích xuất thông tin  
        3️⃣ **Nhập câu hỏi về ảnh & nhận câu trả lời** hoặc **Sử dụng tính năng ghi âm để hỏi bằng giọng nói**  
        4️⃣ **Bạn có thể đặt nhiều câu hỏi liên tiếp**  
        
        ### 🎤 Hướng dẫn sử dụng tính năng giọng nói:
        - Chuyển sang tab "Hỏi đáp giọng nói"
        - Nhấn nút "Bắt đầu ghi âm" và nói câu hỏi của bạn
        - Hoặc sử dụng tính năng ghi âm trực tiếp của Gradio
        - Hệ thống sẽ tự động trả lời bằng giọng nói
        """)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
