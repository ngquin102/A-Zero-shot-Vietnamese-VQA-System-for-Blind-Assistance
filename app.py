
from PIL import Image
from io import BytesIO
import numpy as np
from src.utils import get_device
from src.image_processor import load_image
from src.model_loader import ImageQAModel
from src.speech_utils import SpeechProcessor

class ImageQASystem:
    def __init__(self, device=None, gemini_api_key=None, language="vi"):
        self.device = device or get_device()
        self.model = ImageQAModel(device=self.device, gemini_api_key=gemini_api_key)
        self.model.load_vision_model()
        self.current_image = None
        self.speech_processor = SpeechProcessor(language=language)

    def process_image(self, image_path, max_num=4): #xu ly hinh anh 
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
    
    def record_and_transcribe(self, duration=5): 
        if self.current_image is None:
            return "Vui lòng tải ảnh lên trước!", None, None
        
        audio_np = self.speech_processor.record_audio(duration)
        question = self.speech_processor.transcribe(audio_np)
        
        if "không nhận dạng được" in question.lower():
            return "Không nhận dạng được giọng nói. Vui lòng thử lại.", None, None
            
        answer = self.answer_question(question)
        audio = self.speech_processor.text_to_speech(answer)
        audio_bytes = self.speech_processor.get_audio_bytes(audio)
        self.speech_processor.play_audio(audio)
        
        return f"Câu hỏi: {question}\n\nTrả lời: {answer}", question, audio_bytes

    def process_audio_input(self, audio_input):
        if self.current_image is None:
            return "Vui lòng tải ảnh lên trước!", None, None
        
        try:
            sr, audio_data = audio_input
            audio_np = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
            
            question = self.speech_processor.transcribe(audio_np)
            answer = self.answer_question(question)
            audio = self.speech_processor.text_to_speech(answer)
            audio_bytes = self.speech_processor.get_audio_bytes(audio)
            
            return f"Câu hỏi: {question}\n\nTrả lời: {answer}", question, audio_bytes
        except Exception as e:
            return f"Lỗi xử lý âm thanh: {str(e)}", None, None
