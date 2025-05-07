import os
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from network import ClassifierNetwork
from config import CATEGORIES, IMAGE_FOLDER, IMAGE_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierNetwork(num_classes=len(CATEGORIES)).to(device)
model.load_state_dict(torch.load("Fruit-Classifier/Training/Model_Checkpoint.pth"))
model.eval()

for idx, img in enumerate(os.listdir(IMAGE_FOLDER)):
    try:
        img_array = cv2.imread(os.path.join(IMAGE_FOLDER, img), cv2.IMREAD_COLOR) # Load image in BGR format
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Convert image to RGB format

        img_array_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) # Resize image to img_size x img_size

        X = torch.from_numpy(img_array_resized).to(device) # [img_size, img_size, num_channels]
        X = (X.permute(2, 0, 1).float() / 255.0).unsqueeze(dim=0) # [1, num_channels, img_size, img_size]

        with torch.no_grad():
            logits = model(X).squeeze() # [1, num_classes] -> [num_classes]
            probs = F.softmax(logits, dim=0) # [num_classes]

            pred = probs.argmax(dim=0).item()

            pred_category = CATEGORIES[pred]
            confidence = probs[pred].item() * 100

        print(f"Index: {idx + 1} - Prediction: {pred_category} - Confidence: {confidence:.2f}%")

        # Display the image with the prediction as the title
        plt.figure(figsize=(10, 5))
        plt.imshow(img_array)
        plt.title(f"Prediction: {pred_category} - Confidence: {confidence:.2f}%")
        plt.axis("off")
        plt.savefig(f"Fruit-Classifier/Predictions/{img[:img.index(".")]}_Prediction.png")
        plt.show()

    except:
        pass
