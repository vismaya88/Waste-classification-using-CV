#dataset_loader.py
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = []
        class_to_idx = {}
        idx = 0
        # Traverse all subfolders (biodegradable -> food_waste, etc.)
        for parent in sorted(os.listdir(directory)):
            parent_path = os.path.join(directory, parent)
            if not os.path.isdir(parent_path):
                continue
            for subfolder in sorted(os.listdir(parent_path)):
                full_path = os.path.join(parent_path, subfolder)
                if os.path.isdir(full_path):
                    class_name = f"{parent}/{subfolder}"  # e.g., biodegradable/food_waste
                    classes.append(class_name)
                    class_to_idx[class_name] = idx
                    idx += 1
        return classes, class_to_idx

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = CustomImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes

    return train_loader, val_loader, class_names
