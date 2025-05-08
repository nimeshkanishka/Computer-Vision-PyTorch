import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from network import ClassifierNetwork
from config import NUM_CLASSES, LABEL_TO_CHAR, IMAGE_SIZE, IMAGE_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("Handwritten-Character-Classification/Training/Model_Checkpoint.pth",
                        weights_only=False)

# Create model instance, load the saved weights and set to evaluation mode
model = ClassifierNetwork(IMAGE_SIZE, 1, NUM_CLASSES).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Mean and std of the training dataset (for normalization of inference images)
mean = checkpoint["training_dataset_mean"]
std = checkpoint["training_dataset_std"]

images, X = [], []

for idx, img in enumerate(os.listdir(IMAGE_FOLDER)):
    try:
        img_array = cv2.imread(os.path.join(IMAGE_FOLDER, img), cv2.IMREAD_GRAYSCALE) # Load image in grayscale format

        images.append(img_array.copy()) # This will be used for displaying the images

        img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) # Resize image to IMAGE_SIZE x IMAGE_SIZE

        X.append(img_array.copy()) # This will be used to get predictions from the model

    except Exception as e:
        print(f"Error processing image '{os.path.join(IMAGE_FOLDER, img)}': {e}")

        continue

# Numpy array: (num_images, img_size, img_size, num_channels)
#              Normalized to [0, 1]
X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32) / 255.0

# PyTorch tensor: [num_images, num_channels, img_size, img_size]
#                 Normalized to z-scores
X = (torch.from_numpy(X).permute(0, 3, 1, 2) - mean) / std

batch_index = 0

for batch_start in range(0, X.shape[0], 8):
    batch_end = (batch_start + 8) if (batch_start + 8) < X.shape[0] else X.shape[0]

    # [batch_size, num_channels, img_size, img_size]
    X_batch = X[batch_start : batch_end].to(device)

    with torch.no_grad():
        # [batch_size, num_classes]
        probs = F.softmax(model(X_batch), dim=1)

        # [batch_size]
        preds = probs.argmax(dim=1)

    batch_index += 1

    # Display each image with the predicted label and confidence
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(8):
        if i < (batch_end - batch_start):
            axes[i].imshow(images[batch_start + i], cmap="gray")

            pred_label = LABEL_TO_CHAR[preds[i].item()]
            confidence = probs[i, preds[i].item()].item() * 100

            axes[i].set_title(f"Prediction: {pred_label}\nConfidence: {confidence:.2f}%")

            print(f"Index: {batch_start + i + 1} - Prediction: {pred_label} - Confidence: {confidence:.2f}%")

        # Remove the axes even if there is no image in the subplot
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"Handwritten-Character-Classification/Predictions/Predictions_Batch_{batch_index}.png")
    plt.show()