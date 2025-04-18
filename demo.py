import gradio as gr
import os
from PIL import Image
from io import BytesIO
from src.utils import get_device
from src.image_processor import load_image
from src.model_loader import ImageQAModel

GEMINI_API_KEY = "AIzaSyDHcc-Mme3on7LAx0QxjxTo3M1Fp_SkCrU"


def image_to_bytes(img):
    
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

class ImageQASystem:
    def __init__(self, device=None, gemini_api_key=None):
        self.device = device or get_device()
        self.model = ImageQAModel(device=self.device, gemini_api_key=gemini_api_key)
        self.model.load_vision_model()
        self.current_image = None

    def process_image(self, image_path, max_num=4):
        if not isinstance(image_path, str):
            return "Lỗi: Đầu vào không phải là đường dẫn hợp lệ.", None

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            self.current_image = img

            pixel_values = load_image(img, max_num=max_num)
            caption = self.model.generate_caption(pixel_values)

        return caption, img

    def answer_question(self, question):
        if self.current_image is None:
            return "Vui lòng tải ảnh lên trước!"
        return self.model.answer_question(question)
    
qa_system = ImageQASystem(gemini_api_key=GEMINI_API_KEY)


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ Hệ Thống Zero-shot hỏi đáp nội dung trong hình ảnh")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(label="📂 Tải ảnh từ máy tính", type="filepath")
                caption_output = gr.Textbox(label="📜 Thông tin trích xuất từ ảnh", lines=4)
                image_display = gr.Image(label="🖼️ Ảnh đã tải lên", type="pil")

                with gr.Row():
                    upload_btn = gr.Button("📸 Xử Lý Ảnh")
                    clear_btn = gr.Button("🗑️ Xóa")

            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Hội Thoại Về Ảnh", height=400)
                msg = gr.Textbox(label="📝 Nhập câu hỏi về ảnh")

                with gr.Row():
                    submit_btn = gr.Button("🚀 Gửi")
                    clear_chat_btn = gr.Button("🗑️ Xóa Hội Thoại")

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

        clear_btn.click(lambda: ("", None), outputs=[caption_output, image_display])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])

        gr.Markdown("""
        ## 📖 Hướng Dẫn Sử Dụng
        1️⃣ **Chọn ảnh từ máy tính**  
        2️⃣ **Nhấn "Xử Lý Ảnh"** để trích xuất thông tin  
        3️⃣ **Nhập câu hỏi về ảnh & nhận câu trả lời**  
        4️⃣ **Bạn có thể đặt nhiều câu hỏi liên tiếp**  
        """)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
