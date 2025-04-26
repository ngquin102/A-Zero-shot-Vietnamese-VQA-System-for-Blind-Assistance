import os
import sys
import time
import numpy as np
from PIL import Image
import whisper
import sounddevice as sd
from queue import Queue
import threading
from rich.console import Console
from gtts import gTTS
import tempfile
from pydub import AudioSegment
from pydub.playback import play

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_processor import load_image
from src.model_loader import ImageQAModel
from src.utils import get_device

console = Console()

GEMINI_API_KEY = "......"  

class VoiceAssistant:
    def __init__(self, language="vi"):
        self.language = language
        self.device = get_device()
        self.model = ImageQAModel(device=self.device, gemini_api_key=GEMINI_API_KEY)
        self.model.load_vision_model()

        # Whisper large-v3 trên CPU
        console.print("[green]Đang tải mô hình Whisper large-v3 (CPU)...")
        self.whisper_model = whisper.load_model("large-v3", device="cpu")

        self.current_image = None
        
    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            self.current_image = image
            pixel_values = load_image(image, max_num=4)
            caption = self.model.generate_caption(pixel_values)
            return caption
        except Exception as e:
            return f"Lỗi khi tải ảnh: {str(e)}"
    
    def record_audio(self, duration=5):
        console.print("[yellow]Đang lắng nghe...", end="")
        data_queue = Queue()
        
        def callback(indata, frames, time, status):
            if status:
                console.print(f"[red]{status}")
            data_queue.put(bytes(indata))
        
        with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
            for i in range(duration):
                time.sleep(1)
                console.print(".", end="")
            console.print("")
        
        audio_data = b"".join(list(data_queue.queue))
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_np
    
    def transcribe(self, audio_np):
        with console.status("[cyan]Đang nhận dạng giọng nói...", spinner="dots"):
            result = self.whisper_model.transcribe(audio_np, language=self.language, fp16=False)
            text = result["text"].strip()
        return text
    
    def answer_question(self, question):
        if self.current_image is None:
            return "Vui lòng tải ảnh trước khi đặt câu hỏi."
        
        with console.status("[cyan]Đang xử lý câu trả lời...", spinner="dots"):
            answer = self.model.answer_question(question)
        return answer
    
    def text_to_speech(self, text):
        with console.status("[cyan]Đang chuyển đổi thành giọng nói...", spinner="dots"):
            tts = gTTS(text=text, lang=self.language)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            tts.save(temp_filename)
            audio = AudioSegment.from_mp3(temp_filename)
            os.remove(temp_filename)
        return audio
    
    def play_audio(self, audio):
        play(audio)

def main():
    console.print("[bold cyan]===== Hệ Thống Hỏi Đáp Hình Ảnh Hỗ Trợ Người Khiếm Thị với Zero-Shot Learning =====")
    
    assistant = VoiceAssistant()
    
    image_path = console.input("[bold yellow]Nhập đường dẫn đến file ảnh: ")
    
    console.print("[cyan]Đang tải và xử lý ảnh...")
    caption = assistant.load_image(image_path)
    
    console.print(f"[green]Thông tin trích xuất từ ảnh:[/green]\n{caption}\n")
    
    while True:
        console.print("\n[bold yellow]Nhấn Enter để đặt câu hỏi bằng giọng nói (hoặc gõ 'exit' để thoát)")
        command = console.input()
        
        if command.lower() == 'exit':
            break
        
        audio_np = assistant.record_audio(duration=5)
        question = assistant.transcribe(audio_np)
        console.print(f"[bold yellow]Câu hỏi của bạn:[/bold yellow] {question}")
        
        answer = assistant.answer_question(question)
        console.print(f"[bold green]Câu trả lời:[/bold green] {answer}")
        
        audio = assistant.text_to_speech(answer)
        assistant.play_audio(audio)
    
    console.print("[bold cyan]===== Kết thúc phiên hỏi đáp =====")

if __name__ == "__main__":
    main()
