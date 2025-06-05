#realtime_camera.py

import torch
import cv2
from torchvision import transforms, models
import torch.nn as nn
from dataset_loader import get_data_loaders
from PIL import Image
import numpy as np
from collections import deque
import os
import time

def webcam_predict(model_path, data_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)

    # Load model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing transformations
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and resize bin icons
    biodegradable_bin = cv2.resize(cv2.imread("images/biodegradable_bin.png"), (100, 100))
    non_biodegradable_bin = cv2.resize(cv2.imread("images/non_biodegradable_bin.png"), (100, 100))

    cap = cv2.VideoCapture(0)

    # Setup for prediction stability
    prediction_history = deque(maxlen=10)
    confirmed_label = None
    confirmed_type = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better display
        frame = cv2.resize(frame, (800, 600))  # Adjust the window size here (800x600 for example)

        h, w, _ = frame.shape

        # Define bounding box in center
        box_size = 200
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Crop region of interest (ROI) and preprocess
        roi = frame[y1:y2, x1:x2]
        image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            label = class_names[pred.item()]
            label_clean = label.strip().lower()

        # Stability logic: Keep history of predictions
        prediction_history.append(label)
        most_common = max(set(prediction_history), key=prediction_history.count)

        # Dynamically update label and type every frame
        confirmed_label = most_common  # Use most common label (from prediction history)

        # Match 'biodegradable' or 'non-biodegradable' exactly
        if "biodegradable" in confirmed_label.lower() and "non_biodegradable" not in confirmed_label.lower():
            confirmed_type = "Biodegradable"
        elif "non_biodegradable" in confirmed_label.lower():
            confirmed_type = "Non-Biodegradable"
        else:
            confirmed_type = "Unknown"  # This case handles any other categories that might not be biodegradable

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if confirmed_label:
            # Dynamic label placement with clipping
            label_text = f"Detected: {confirmed_label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Measure text size
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Adjust position to avoid overflow
            text_x = x1
            if x1 + text_width > frame.shape[1]:
                text_x = frame.shape[1] - text_width - 10  # shift left to fit

            text_y = max(y1 - 10, text_height + 10)

            # Draw background rectangle
            cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                                (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)

            # Put label text on top
            cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

            # Place bin image near detected object based on Type
            bin_width, bin_height = 100, 100  # Bin image dimensions
            if confirmed_type == "Biodegradable":
                # Ensure the bin icon is within frame bounds
                x_bin = max(x1 - bin_width - 10, 0)  # Add margin
                y_bin = min(y1, h - bin_height)      # Ensure the image fits vertically
                frame[y_bin:y_bin+bin_height, x_bin:x_bin+bin_width] = biodegradable_bin
            elif confirmed_type == "Non-Biodegradable":
                x_bin = min(x2 + 10, w - bin_width)  # Add margin
                y_bin = min(y1, h - bin_height)      # Ensure the image fits vertically
                frame[y_bin:y_bin+bin_height, x_bin:x_bin+bin_width] = non_biodegradable_bin

        # Show the updated frame
        cv2.imshow("Real-Time Waste Classification", frame)

        # Capture and save the cropped image when the user presses the 'c' key
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Save only the cropped object (ROI), not the whole frame
            image_name = f"captured_object_{int(time.time())}.jpg"
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, roi)  # Save the cropped region (ROI) here
            print(f"Image saved to {save_path}")

        # Close the camera feed with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_predict("model_save/waste_model.pt", "Dataset", "static/uploads")
