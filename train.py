# train.py
import os
import torch
import torch.nn as nn
from torchvision import models
from dataset_loader import get_data_loaders
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.makedirs("model_save", exist_ok=True)

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def train_and_save(model_name, data_dir, num_epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)

    model = get_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    save_path = f"model_save/{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved {model_name} to {save_path}")

if __name__ == "__main__":
    data_dir = "Dataset"
    models_to_train = ["resnet18", "vgg16", "efficientnet_b0", "alexnet"]
    for model_name in models_to_train:
        train_and_save(model_name, data_dir)
