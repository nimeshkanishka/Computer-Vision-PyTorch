# Folder containing the dataset
DATA_FOLDER = r"D:\Datasets\English-Handwritten-Characters-Dataset"

# Number of classes in the dataset
NUM_CLASSES = 62

# Label to character mapping for the dataset
LABEL_TO_CHAR = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Number of images available per class in the dataset
NUM_IMAGES_PER_CLASS = 55

# Index at which training data of a class ends
TRAIN_END = 45

# Index at which validation data of a class ends
VALIDATION_END = 50

# Target size to which each image will be resized (IMAGE_SIZE x IMAGE_SIZE pixels)
IMAGE_SIZE = 32

# Number of images processed in each training or validation batch
BATCH_SIZE = 32

# Number of times the entire training dataset is passed through the model
NUM_EPOCHS = 25

# Folder containing external images to classify using the trained model (used during inference)
IMAGE_FOLDER = r"D:\Python Projects\Computer Vision PyTorch\Handwritten-Character-Classification\Images"