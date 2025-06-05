#predict_image.py
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from dataset_loader import get_data_loaders

def predict_image(image_path, model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, class_names = get_data_loaders(data_dir)
    
    print(f"Class Names Used: {class_names}")

    num_classes = len(class_names)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    print(f"🔍 Predicted: {class_names[predicted.item()]}")
    return class_names[predicted.item()]

if __name__ == "__main__":
    predict_image("Dataset/val/biodegradable/food_waste/2_178.jpg", "model_save/waste_model.pt", "Dataset")
