import os
import random
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 64
DATA_FOLDER = r"D:\Datasets\kaggle_cats_and_dogs"
CATEGORIES = ["Cat", "Dog"]

dataset = []
for label, category in enumerate(CATEGORIES):
    img_folder = os.path.join(DATA_FOLDER, category)
    num_images = 0

    for img in tqdm(os.listdir(img_folder), desc=f"Processing '{category}'"):
        try:
            img_array = cv2.imread(os.path.join(img_folder, img), cv2.IMREAD_COLOR) # Load image in BGR format
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Convert image to RGB format
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # Resize image to img_size x img_size

            dataset.append((img_array, label))

            num_images += 1

        except:
            pass

    print(f"----- {num_images} images added to the dataset from '{category}' folder -----")

random.shuffle(dataset)

X, y = [], []
for features, label in dataset:
    X.append(features)
    y.append(label)

X = np.array(X) # (num_images, img_size, img_size, num_channels)
y = np.array(y) # (num_images,)

np.savez("Cat-Dog-Classifier/Data/cat_dog_dataset.npz", X=X, y=y)