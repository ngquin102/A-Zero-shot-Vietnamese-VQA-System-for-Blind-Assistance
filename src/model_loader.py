import torch
import os
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai

class ImageQAModel:
    def __init__(self, device='cpu', gemini_api_key=None):
        self.device = device
        self.vision_model = None
        self.vision_tokenizer = None
        self.caption = None
        gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("Bạn cần cung cấp Gemini API Key qua biến môi trường GOOGLE_API_KEY hoặc đối số.")
        
        genai.configure(api_key=gemini_api_key)
        self.llm_model = genai.GenerativeModel("gemini-1.5-pro")

    def load_vision_model(self, model_name="5CD-AI/Vintern-1B-v3_5"):
        """Load vision-language model."""
        try:
            self.vision_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False
            ).eval().to(self.device)
        except:
            self.vision_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().to(self.device)

        self.vision_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    def generate_caption(self, pixel_values):
        """Trích xuất toàn bộ thông tin trong ảnh."""
        if self.vision_model is None:
            raise ValueError("Vision model not loaded. Call load_vision_model() first.")

        generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=2, repetition_penalty=3.5)
        question = "<image>\nTrích xuất toàn bộ thông tin trong ảnh"

        pixel_values = pixel_values.to(self.device)
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)

        caption = self.vision_model.chat(self.vision_tokenizer, pixel_values, question, generation_config)
        self.caption = caption
        return caption

    def answer_question(self, question):
        """Trả lời câu hỏi dựa trên nội dung đã trích xuất từ ảnh."""
        if self.caption is None:
            raise ValueError("No caption available. Process an image first.")

        prompt = f"""Văn bản từ hình ảnh: {self.caption}\nCâu hỏi: {question}\nTrả lời:"""

        response = self.llm_model.generate_content(prompt)
        return response.text.strip()
