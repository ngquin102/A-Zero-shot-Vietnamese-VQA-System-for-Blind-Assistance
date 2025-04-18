import requests
from io import BytesIO
from PIL import Image
from src.image_processor import load_image
from src.model_loader import ImageQAModel
from src.utils import get_device, pil_to_bytes

class ImageQASystem:
    def __init__(self, device=None, gemini_api_key=None):
        self.device = device or get_device()
        self.model = ImageQAModel(device=self.device, gemini_api_key=gemini_api_key)
        self.model.load_vision_model()

    def process_image(self, image_path=None, image_url=None, max_num=4):
        """Xử lý ảnh từ file hoặc URL và trích xuất thông tin."""
        
        if image_path:
            img = Image.open(image_path).convert("RGB")
        elif image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return "Vui lòng tải lên hoặc nhập URL ảnh.", None

        self.model.current_image = img  

        pixel_values = load_image(img, max_num=max_num)
        caption = self.model.generate_caption(pixel_values)  
        return caption, pil_to_bytes(img)

    def answer_question(self, question):
        """Trả lời câu hỏi về ảnh đã xử lý."""
        return self.model.answer_question(question)
