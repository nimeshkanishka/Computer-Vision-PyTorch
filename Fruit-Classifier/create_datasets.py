import os
import cv2
import numpy as np
from tqdm import tqdm
from config import DATA_FOLDER, DATASETS, CATEGORIES, IMAGE_SIZE

for dataset in DATASETS:
    dataset_folder = os.path.join(DATA_FOLDER, dataset)
    X, y = [], []

    for label, category in enumerate(CATEGORIES):
        img_folder = os.path.join(dataset_folder, category)

        for img in tqdm(os.listdir(img_folder), desc=f"Processing '{dataset}/{category}'"):
            try:
                img_array = cv2.imread(os.path.join(img_folder, img), cv2.IMREAD_COLOR) # Load image in BGR format
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Convert image to RGB format
                img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) # Resize image to img_size x img_size

                X.append(img_array)
                y.append(label)

            except:
                pass

    X = np.array(X) # (num_images, img_size, img_size, num_channels)
    y = np.array(y) # (num_images,)

    np.savez(f"Fruit-Classifier/Data/{dataset}-Dataset.npz", X=X, y=y)

    print(f"----- {dataset} dataset saved to 'Fruit-Classifier/Data/{dataset}-Dataset.npz' -----")