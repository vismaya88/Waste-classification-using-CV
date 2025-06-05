#evaluate_model.py
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_loader import get_data_loaders
from torchvision import models
import torch.nn as nn
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def evaluate_model(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # <-- ADD Accuracy inside the function
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model('model_save/waste_model.pt', 'Dataset')
