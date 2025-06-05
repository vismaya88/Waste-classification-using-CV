#app.py
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
from dataset_loader import get_data_loaders

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names (assuming same function as training)
_, _, class_names = get_data_loaders('Dataset')
num_classes = len(class_names)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ResNet-18 model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model_save/waste_model.pt', map_location=device))
model = model.to(device)
model.eval()

# Define a function to predict the image
def predict_image(image_path):
    # Define the preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the image
    img = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format

    # Apply the transformation
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Get prediction from the uploaded image
    prediction = predict_image(filepath)

    # Debugging: Print prediction to verify
    print(f"Prediction: {prediction}")

    # Ensure prediction follows 'parent_class/sub_class' format
    try:
        main_cat, sub_cat = prediction.split('/')
    except ValueError:
        return "Prediction format is incorrect", 500

    # Determine the bin type
    bin_type = "Biodegradable" if main_cat == 'biodegradable' else "Non-Biodegradable"

    return render_template('result.html',
                           prediction=prediction,
                           bin_type=bin_type,
                           image_file=filename)

if __name__ == '__main__':
    app.run(debug=True)
