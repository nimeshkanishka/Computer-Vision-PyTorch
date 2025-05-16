import os
import math
import random
import cv2
import numpy as np
from config import DATA_FOLDER, CLASSES, IMAGE_SIZE, RGB


def resize_image(image_array):    
    # Resize the image preserving aspect ratio (shorter side becomes new_size pixels)
    # image_array will have 3 dims (if the image is loaded in RGB format) or 2 dims (if the image is loaded in grayscale format)
    if len(image_array.shape) == 3:
        h, w, _ = image_array.shape
    else:
        h, w = image_array.shape
    factor = IMAGE_SIZE / min(h, w)
    h, w = math.ceil(h * factor), math.ceil(w * factor)
    image_array = cv2.resize(image_array, (w, h), interpolation=cv2.INTER_AREA)

    # Retain IMAGE_SIZE x IMAGE_SIZE part from the center of the image
    x_start = (w - IMAGE_SIZE) // 2
    y_start = (h - IMAGE_SIZE) // 2
    image_array = image_array[y_start : y_start + IMAGE_SIZE, x_start : x_start + IMAGE_SIZE]

    return image_array


def rotate_image(image_array, max_angle):
    # Rotate image by a random angle between -(max_angle) and +(max_angle) degrees
    rot_angle = random.uniform(-max_angle, max_angle)
    rot_matrix = cv2.getRotationMatrix2D(center=(IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                                         angle=rot_angle, scale=1.0)
    
    return cv2.warpAffine(image_array, rot_matrix, (IMAGE_SIZE, IMAGE_SIZE),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def adjust_image_contrast_and_brightness(image_array, max_contrast_shift, max_brightness_shift):
    # Contrast adjustment
    alpha = random.uniform(1 - max_contrast_shift, 1 + max_contrast_shift)
    # Brightness adjustment
    beta = random.uniform(-max_brightness_shift, max_brightness_shift)

    return cv2.convertScaleAbs(image_array, alpha=alpha, beta=beta)


"""
Create Training Dataset
"""
dataset_folder = os.path.join(DATA_FOLDER, "Training")

# Make sure equal numbers of images are added to the dataset from each class. As Chest X-Ray Images dataset has
# significantly more images in the class "PNEUMONIA" than in the class "NORMAL", we loop over the images in the
# class "NORMAL" multiple times and add a version of each image preprocessed differently to the dataset in each
# pass until we have added the same number of images to the dataset from the class "NORMAL" as we will add to it
# from the class "PNEUMONIA" in one pass.
num_images_per_class = max([len(os.listdir(os.path.join(dataset_folder, img_class))) for img_class in CLASSES])

X, y = [], []

for label, img_class in enumerate(CLASSES):
    img_folder = os.path.join(dataset_folder, img_class)
    img_files = os.listdir(img_folder)
    # Number of images currently added to the dataset from this class
    num_images = 0

    while num_images < num_images_per_class:
        # Number of more images to be added to the dataset from this class
        num_images_to_add = num_images_per_class - num_images

        # Shuffle the images in every pass as we do not want to add them to the dataset in the same order in
        # every pass (if there is more than one pass).
        random.shuffle(img_files)

        for img_file in img_files[:num_images_to_add if num_images_to_add < len(img_files) else len(img_files)]:            
            try:
                # Image Preprocessing
                # Load the image
                if RGB:
                    img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_COLOR)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # (height, width, 3)
                else:
                    img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_GRAYSCALE) # (height, width)

                # Resize image to IMAGE_SIZE x IMAGE_SIZE
                img_array = resize_image(img_array)
                
                # Rotate the image by a random angle between -10 and +10 degrees
                img_array = rotate_image(img_array, max_angle=10)
                
                # Adjust contrast and brightness by upto 15%
                img_array = adjust_image_contrast_and_brightness(img_array,
                                                                 max_contrast_shift=0.15,
                                                                 max_brightness_shift=15)

                X.append(img_array)
                y.append(label)

                num_images += 1

            except Exception as e:
                print(f"Error processing image '{os.path.join(img_folder, img_file)}': {e}")

                continue

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3 if RGB else 1) # (num_images, img_size, img_size, num_channels)
y = np.array(y) # (num_images,)

np.savez(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Training-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz", X=X, y=y)

print(f"----- Training dataset saved to 'Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Training-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz' -----")


"""
Create Validation Dataset
"""
dataset_folder = os.path.join(DATA_FOLDER, "Validation")

X, y = [], []

for label, img_class in enumerate(CLASSES):
    img_folder = os.path.join(dataset_folder, img_class)

    for img_file in os.listdir(img_folder):
        try:
            # Image Preprocessing
            # Load the image
            if RGB:
                img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # (height, width, 3)
            else:
                img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_GRAYSCALE) # (height, width)

            # Resize image to IMAGE_SIZE x IMAGE_SIZE
            img_array = resize_image(img_array)

            X.append(img_array)
            y.append(label)

        except Exception as e:
            print(f"Error processing image '{os.path.join(img_folder, img_file)}': {e}")

            continue

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3 if RGB else 1) # (num_images, img_size, img_size, num_channels)
y = np.array(y) # (num_images,)

np.savez(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Validation-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz", X=X, y=y)

print(f"----- Validation dataset saved to 'Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Validation-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz' -----")


"""
Create Test Dataset
"""
dataset_folder = os.path.join(DATA_FOLDER, "Test")

X, y = [], []

for label, img_class in enumerate(CLASSES):
    img_folder = os.path.join(dataset_folder, img_class)

    for img_file in os.listdir(img_folder):
        try:
            # Image Preprocessing
            # Load the image
            if RGB:
                img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # (height, width, 3)
            else:
                img_array = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_GRAYSCALE) # (height, width)

            # Resize image to IMAGE_SIZE x IMAGE_SIZE
            img_array = resize_image(img_array)

            X.append(img_array)
            y.append(label)

        except Exception as e:
            print(f"Error processing image '{os.path.join(img_folder, img_file)}': {e}")

            continue

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3 if RGB else 1) # (num_images, img_size, img_size, num_channels)
y = np.array(y) # (num_images,)

np.savez(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Test-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz", X=X, y=y)

print(f"----- Test dataset saved to 'Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Test-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz' -----")