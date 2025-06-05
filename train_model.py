#train_model.py

import torch
import torch.nn as nn
from torchvision import models
from dataset_loader import get_data_loaders

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_model(data_dir, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'model_save/waste_model.pt')
    print("✅ Model trained and saved as waste_model.pt")
    return model, class_names

if __name__ == "__main__":
    train_model('Dataset')
