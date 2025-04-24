import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the same model architecture used in training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = Net()
model.load_state_dict(torch.load("model/digit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Streamlit app
st.title("🖌️ Draw a Digit - Recognizer")
st.write("Draw a digit (0 to 9) below and click **Predict** to see what the model thinks!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black
    stroke_width=10,
    stroke_color="#FFFFFF",  # White stroke
    background_color="#000000",  # Black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))  # Take one channel
    img = img.convert("L")  # Convert to grayscale
    
    st.image(img, caption="Your Drawing (resized for model)", width=140)
    
    if st.button("Predict"):
        img_transformed = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_transformed)
            prediction = torch.argmax(output, 1).item()
        import random

        emojis = ["🎯", "🚀", "🤖", "🧠", "✨", "🎉", "🔥", "📈", "🔢"]
        confetti = random.choice(emojis)

        st.balloons()  # Streamlit's built-in confetti animation
        st.markdown(f"""
        ### 🧠 Predicted Digit:  
        # <span style='color:purple'>{prediction} {confetti}</span>
        """, unsafe_allow_html=True)

