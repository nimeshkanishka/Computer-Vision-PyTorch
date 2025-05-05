import os
import torch
import cv2
import matplotlib.pyplot as plt
from network import ClassifierNetwork

IMG_SIZE = 64 # This must match the size used in create_dataset.py
IMAGE_FOLDER = r"D:\Python Projects\Computer Vision PyTorch\Cat-Dog-Classifier\Images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierNetwork().to(device)
model.load_state_dict(torch.load("Cat-Dog-Classifier/Training/Model_Checkpoint.pth"))
model.eval()

for idx, img in enumerate(os.listdir(IMAGE_FOLDER)):
    try:
        img_array = cv2.imread(os.path.join(IMAGE_FOLDER, img), cv2.IMREAD_COLOR) # Load image in BGR format
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Convert image to RGB format

        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # Resize image to img_size x img_size

        X = torch.from_numpy(img_array_resized).to(device) # [img_size, img_size, num_channels]
        X = (X.permute(2, 0, 1).float() / 255.0).unsqueeze(dim=0) # [1, num_channels, img_size, img_size]

        with torch.no_grad():
            pred = model(X).squeeze().item()
            
        if pred < 0.5:
            pred_category = "Cat"
            confidence = 1 - pred
        else:
            pred_category = "Dog"
            confidence = pred

        print(f"Index: {idx + 1} - Prediction: {pred_category} - Confidence: {(confidence * 100):.2f}%")

        # Display the image with the prediction as the title
        plt.imshow(img_array)
        plt.title(f"Prediction: {pred_category} - Confidence: {(confidence * 100):.2f}%")
        plt.axis("off")
        plt.show()

    except:
        pass