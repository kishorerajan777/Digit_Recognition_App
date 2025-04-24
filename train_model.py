import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Optional: for progress bar

# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)  # Updated to match the size after conv layers
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Set device (using CPU in your case)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, optimizer, and loss function
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Ensure the 'model' directory exists, otherwise create it
if not os.path.exists("model"):
    os.makedirs("model")
    print("Model directory created.")

# Train the model
for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100):  # Progress bar
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Save the trained model
model_save_path = "model/digit_model.pth"
torch.save(model.state_dict(), model_save_path)  # Save only the model parameters
print(f"Model saved at {model_save_path}")
