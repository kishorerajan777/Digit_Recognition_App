from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

def preprocess_image(image):
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    return image
