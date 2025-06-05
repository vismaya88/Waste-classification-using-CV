#evaluate.py
import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataset_loader import get_data_loaders
import joblib
import warnings
from PIL import ImageFile  # <- Add this
import seaborn as sns
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True  # <- Add this

warnings.filterwarnings("ignore")

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    
    #elif model_name == "efficientnet_b0":
        #model = models.efficientnet_b0(pretrained=True)
        #model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def evaluate(model_name, model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)

    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n📊 Classification Report for {model_name.upper()}:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"✅ Accuracy: {acc*100:.2f}%")

# Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {model_name.upper()}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return acc, model

if __name__ == "__main__":
    data_dir = "Dataset"
    model_dir = "model_save"
    valid_models = ["resnet18", "vgg16", #"efficientnet_b0", 
                    "alexnet"]

    best_acc = 0
    best_model_name = None

    for file in os.listdir(model_dir):
        if file.endswith(".pt"):
            model_name = file.replace(".pt", "")
            #if model_name not in valid_models:
                #print(f"⚠️ Skipping unknown model file: {file}")
                #continue

            model_path = os.path.join(model_dir, file)
            try:
                acc, _ = evaluate(model_name, model_path, data_dir)
                if acc > best_acc:
                    best_acc = acc
                    best_model_name = model_name
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")

    if best_model_name:
        print(f"\n🏆 Best Model: {best_model_name} ({best_acc*100:.2f}%)")
        with open("best_model.txt", "w") as f:
            f.write(best_model_name)
    else:
        print("❌ No valid models evaluated.")
