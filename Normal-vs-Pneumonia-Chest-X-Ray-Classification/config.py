# Folder containing the dataset
DATA_FOLDER = r"D:\Datasets\Normal-vs-Pneumonia-Chest-X-Ray-Images"

# Image classes
CLASSES = ["NORMAL", "PNEUMONIA"]

# Target size to which each image will be resized (IMAGE_SIZE x IMAGE_SIZE pixels)
IMAGE_SIZE = 256

# Whether the images should be loaded in RGB (True) or grayscale (False) format
RGB = False

# Network architecture to be used for training
# Currently available: Custom, EfficientNet-B0 (Requires 224x224 RGB images)
NETWORK = "Custom"

# Number of images processed in each training, validation or testing batch
BATCH_SIZE = 32

# Number of times the entire training dataset is passed through the model
NUM_EPOCHS = 25

# Folder containing the images to be classified during inference
IMAGE_FOLDER = r"D:\Python Projects\Computer Vision PyTorch\Normal-vs-Pneumonia-Chest-X-Ray-Classification\Images"