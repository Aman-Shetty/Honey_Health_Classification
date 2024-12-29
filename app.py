import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 6)  # Adjust final layer
model.load_state_dict(torch.load('./models/vgg16.pth'))
model.eval()

# Define class labels
class_labels = [
    "Healthy",
    "Few Varroa, Hive Beetles",
    "Varroa, Small Hive Beetles",
    "Ant Problems",
    "Hive Being Robbed",
    "Missing Queen"
]

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Route to serve the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Predict the class
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = probabilities.argmax().item()

    # Return prediction result
    response = {
        "predicted_class": class_labels[predicted_class],
        "class_probabilities": {
            class_labels[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)
        },
        "image_path": f"/static/uploads/{filename}"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
