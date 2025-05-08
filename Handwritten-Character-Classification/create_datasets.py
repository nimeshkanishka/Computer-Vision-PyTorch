import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from config import DATA_FOLDER, LABEL_TO_CHAR, IMAGE_SIZE, NUM_IMAGES_PER_CLASS, TRAIN_END, VALIDATION_END

df = pd.read_csv(os.path.join(DATA_FOLDER, "Img_Labels.csv"))

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

i = 0

for data in tqdm(df.values):
    try:
        img_array = cv2.imread(os.path.join(DATA_FOLDER, data[0]), cv2.IMREAD_GRAYSCALE) # Load image in grayscale format
        img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) # Resize image to IMAGE_SIZE x IMAGE_SIZE

        # Determine the dataset to add the image to based on the index
        if i < TRAIN_END:
            X_train.append(img_array)
            y_train.append(LABEL_TO_CHAR.index(data[1])) # As the labels range from 0-9, A-Z and a-z, we need to convert all of them to integers (0-61)
        elif i < VALIDATION_END:
            X_val.append(img_array)
            y_val.append(LABEL_TO_CHAR.index(data[1])) # As the labels range from 0-9, A-Z and a-z, we need to convert all of them to integers (0-61)
        else:
            X_test.append(img_array)
            y_test.append(LABEL_TO_CHAR.index(data[1])) # As the labels range from 0-9, A-Z and a-z, we need to convert all of them to integers (0-61)

        i = (i + 1) % NUM_IMAGES_PER_CLASS

    except Exception as e:
        print(f"Error processing image '{os.path.join(DATA_FOLDER, data[0])}': {e}")

        continue

# Save training dataset as numpy arrays
X_train = np.array(X_train).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) # (num_images, img_size, img_size, num_channels)
y_train = np.array(y_train) # (num_images,)
np.savez("Handwritten-Character-Classification/Data/Training-Dataset.npz", X=X_train, y=y_train)

# Save validation dataset as numpy arrays
X_val = np.array(X_val).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) # (num_images, img_size, img_size, num_channels)
y_val = np.array(y_val) # (num_images,)
np.savez("Handwritten-Character-Classification/Data/Validation-Dataset.npz", X=X_val, y=y_val)

# Save test dataset as numpy arrays
X_test = np.array(X_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) # (num_images, img_size, img_size, num_channels)
y_test = np.array(y_test) # (num_images,)
np.savez("Handwritten-Character-Classification/Data/Test-Dataset.npz", X=X_test, y=y_test)