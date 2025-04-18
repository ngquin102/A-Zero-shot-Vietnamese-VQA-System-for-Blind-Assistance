import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

def plot_image(image_path):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    img = Image.open(image_path) if isinstance(image_path, str) else Image.fromarray(image_path)
    plt.imshow(img)
    plt.axis('off')
    return plt.gcf()

def get_device():
    """Get the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pil_to_bytes(pil_img):
    """Convert PIL Image to bytes for display."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    return buf.getvalue()

def bytes_to_pil(img_bytes):
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))