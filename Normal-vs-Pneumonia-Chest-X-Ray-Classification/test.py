import os
import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from custom_network import CustomClassifierNetwork
from pretrained_networks import EfficientNet_B0, ResNet50
from config import CLASSES, IMAGE_SIZE, RGB, NETWORK, BATCH_SIZE, IMAGE_FOLDER

# PREPROCESS IMAGES
images = []
X = []

for img_file in os.listdir(IMAGE_FOLDER):
    try:
        # Load the image in RGB format for displaying
        img_array = cv2.imread(os.path.join(IMAGE_FOLDER, img_file), cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # (height, width, 3)

        # Store RGB image for displaying
        images.append(img_array)

        h, w, _ = img_array.shape

        # If the network requires grayscale images, convert the image to grayscale format
        if not RGB:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # (height, width)

        # Resize the image preserving aspect ratio (shorter side becomes IMAGE_SIZE pixels)
        factor = IMAGE_SIZE / min(h, w)
        h, w = math.ceil(h * factor), math.ceil(w * factor)
        img_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_AREA)

        # Retain IMAGE_SIZE x IMAGE_SIZE part from the center of the image
        x_start = (w - IMAGE_SIZE) // 2
        y_start = (h - IMAGE_SIZE) // 2
        img_array = img_array[y_start : y_start + IMAGE_SIZE, x_start : x_start + IMAGE_SIZE]

        # Store preprocessed image
        X.append(img_array)

    except Exception as e:
        print(f"Error processing image '{os.path.join(IMAGE_FOLDER, img_file)}': {e}")

        continue

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3 if RGB else 1) # (num_images, img_size, img_size, num_channels)

# Use GPU if available. Otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINE NETWORK
match NETWORK:
    case "Custom":
        model = CustomClassifierNetwork().to(device)
        
        # Channel-wise mean and std of the training images for normalization
        dataset_train = np.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Training-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz")
        mean = torch.from_numpy(dataset_train["mean"]).view(1, -1, 1, 1) # [num_channels] -> [1, num_channels, 1, 1]
        std = torch.from_numpy(dataset_train["std"]).view(1, -1, 1, 1) # [num_channels] -> [1, num_channels, 1, 1]

    case "EfficientNet-B0":
        model = EfficientNet_B0().to(device)
        
        # ImageNet channel-wise mean and std for normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, -1, 1, 1) # [3] -> [1, 3, 1, 1]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, -1, 1, 1) # [3] -> [1, 3, 1, 1]

    case "ResNet-50":
        model = ResNet50().to(device)
        
        # ImageNet channel-wise mean and std for normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, -1, 1, 1) # [3] -> [1, 3, 1, 1]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, -1, 1, 1) # [3] -> [1, 3, 1, 1]

    case _:
        raise ValueError(f"Unsupported network architecture: '{NETWORK}'. Choose from 'Custom', 'EfficientNet-B0' or 'ResNet-50'.")
    
# Load the saved weights and set the network to evaluation mode
model.load_state_dict(torch.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Training/Model-Checkpoint-{NETWORK}.pth"))
model.eval()

# [num_images, img_size, img_size, num_channels] -> [num_images, num_channels, img_size, img_size]
X = (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0 - mean) / std

for batch_start in range(0, X.shape[0], BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE if batch_start + BATCH_SIZE < X.shape[0] else X.shape[0]

    X_batch = X[batch_start:batch_end].to(device)

    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_batch)).squeeze() # [batch_size, 1] -> [batch_size]

    for i in range(batch_end - batch_start):
        if y_pred[i].item() < 0.5:
            pred = CLASSES[0]
            confidence = (1 - y_pred[i].item()) * 100
        else:
            pred = CLASSES[1]
            confidence = y_pred[i].item() * 100

        print(f"Image: {batch_start + i + 1} - Prediction: {pred} - Confidence: {confidence:.2f}%")

        plt.figure(figsize=(5, 5))
        plt.imshow(images[batch_start + i])
        plt.title(f"Prediction: {pred}\nConfidence: {confidence:.2f}%")
        plt.axis("off")
        plt.show()