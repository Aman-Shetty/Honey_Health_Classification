import torch
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image

# Recreate the model architecture
model = vgg16(pretrained=False)  # Load without pretrained weights
model.classifier[6] = torch.nn.Linear(4096, 6)  # Adjust final layer to match 6 classes
model.load_state_dict(torch.load("vgg16.pth"))  # Load weights
model.eval()  # Set model to evaluation mode

# Define the class labels
class_labels = [
    "Healthy",
    "Few Varroa, Hive Beetles",
    "Varroa, Small Hive Beetles",
    "Ant Problems",
    "Hive Being Robbed",
    "Missing Queen"
]

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load your custom image
image_path = "001_118.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict using the model
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class
predicted_class = probabilities.argmax().item()

# Print the result
print(f"Predicted class: {class_labels[predicted_class]}")
print("Class probabilities:")
for i, prob in enumerate(probabilities.tolist()):
    print(f"  {class_labels[i]}: {prob:.4f}")
