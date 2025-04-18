import torch
from PIL import Image
import warnings
import time
import os

from image_processor import load_image
from model_loader import ImageQAModel  



os.environ["GOOGLE_API_KEY"] = "....."  


def main():
    warnings.filterwarnings("ignore")

    image_path = "/home/quynh/quynh/My_D/CODE/PythonProject_HAUI/Đồ án tốt nghiệp/zero_shot_vqa/a.png"
    
    print("Initializing ImageQAModel...")
    model = ImageQAModel(device='cpu')  
    
    print("\nLoading vision model...")
    start_time = time.time()
    try:
        model.load_vision_model()
        print(f"✓ Vision model loaded successfully ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"✗ Failed to load vision model: {e}")
        return

    print("\nLoading image from local path...")
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        model.current_image = image  # Lưu ảnh cho Q&A
        print(f"✓ Image loaded successfully from: {image_path}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return

    print("\nPreprocessing image...")
    try:
        pixel_values = load_image(image, input_size=448, max_num=12)
        print("✓ Image preprocessed successfully")
    except Exception as e:
        print(f"✗ Failed to preprocess image: {e}")
        return

    print("\nExtracting information from image...")
    start_time = time.time()
    try:
        caption = model.generate_caption(pixel_values)
        print(f"✓ Information extracted successfully ({time.time() - start_time:.2f}s)")
        print(f"\n📄 Extracted Info:\n{caption}\n")
    except Exception as e:
        print(f"✗ Failed to extract info: {e}")
        return
    
    print("Testing question answering...")
    test_questions = [
        "Tóm tắt thông tin trong ảnh",
        "Ảnh có những nội dung gì?",
        "Có chữ nào trong ảnh không?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        start_time = time.time()
        try:
            answer = model.answer_question(question)
            print(f"Answer ({time.time() - start_time:.2f}s): {answer}")
        except Exception as e:
            print(f"✗ Failed to answer question: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
