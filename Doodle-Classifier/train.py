import os
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from network import ClassifierNetwork

#################### CONFIG ####################
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################

# Define transforms
# Add random scaling, rotation and offsets to training images
# Random scaling between 70% and 110%
# Random rotation between -15 and +15 degrees
# Random horizontal and vertical shifts by up to 15%
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.2, 0.2),
        scale=(0.7, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

# No augmentations for images used for evaluation (validation and testing) 
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

# Load dataset and apply train and eval transforms
train_dataset_full = datasets.ImageFolder(
    root=r"D:\Datasets\Quick-Draw-10-Categories",
    transform=train_transform
)

eval_dataset_full = datasets.ImageFolder(
    root=r"D:\Datasets\Quick-Draw-10-Categories",
    transform=eval_transform
)

# Split into train, validation and test datasets
labels = [label for _, label in train_dataset_full.samples]
indices = list(range(len(labels)))

# First split the entire dataset into train dataset (80%) and evaluation (validation and test) dataset (20%)
train_idx, eval_idx, train_labels, eval_labels = train_test_split(
    indices, labels,
    test_size=0.2,
    stratify=labels
)

# Then split the evaluation dataset into validation dataset (50% of 20% = 10%) and test dataset (50% of 20% = 10%)
val_idx, test_idx = train_test_split(
    eval_idx,
    test_size=0.5,
    stratify=eval_labels
)

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset = Subset(eval_dataset_full, val_idx)
test_dataset = Subset(eval_dataset_full, test_idx)

# Create train, validation and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define model and optimizer
model = ClassifierNetwork().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to keep track of loss and accuracy at each epoch
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    # Set the model to training mode
    model.train()

    total_loss = 0.0
    correct_preds = 0

    for X_batch, y_batch in tqdm(train_loader):
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        y_pred = model(X_batch)

        loss = F.cross_entropy(y_pred, y_batch)

        total_loss += loss.item()
        correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(train_loader)
    train_loss_history.append(train_loss)
    train_acc = correct_preds / len(train_loader.dataset)
    train_acc_history.append(train_acc)
    print(f"Train loss: {train_loss:.4f} - Train accuracy: {(train_acc * 100):.2f}%")

    # Set the model to evaluation mode (for validation)
    model.eval()

    total_loss = 0.0
    correct_preds = 0
    
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        with torch.no_grad():
            y_pred = model(X_batch)

        total_loss += F.cross_entropy(y_pred, y_batch).item()
        correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

    val_loss = total_loss / len(val_loader)
    val_loss_history.append(val_loss)
    val_acc = correct_preds / len(val_loader.dataset)
    val_acc_history.append(val_acc)
    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {(val_acc * 100):.2f}%")

# Save model weights
os.makedirs("Doodle-Classifier/models", exist_ok=True)
torch.save(model.state_dict(), "Doodle-Classifier/models/model_checkpoint.pth")

# Plot training and validation loss and accuracy graphs
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
epochs = [i for i in range(1, NUM_EPOCHS + 1)]

# Loss vs Epoch graph
axes[0].plot(epochs, train_loss_history, label="Train Loss")
axes[0].plot(epochs, val_loss_history, label="Validation Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy vs Epoch graph
axes[1].plot(epochs, train_acc_history, label="Train Accuracy")
axes[1].plot(epochs, val_acc_history, label="Validation Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

os.makedirs("Doodle-Classifier/training", exist_ok=True)
plt.savefig("Doodle-Classifier/training/loss_accuracy_vs_epoch.png")
plt.show()

# Testing
model.eval()

total_loss = 0.0
correct_preds = 0

for X_batch, y_batch in test_loader:
    X_batch = X_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    with torch.no_grad():
        y_pred = model(X_batch)

    total_loss += F.cross_entropy(y_pred, y_batch).item()
    correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

test_loss = total_loss / len(test_loader)
test_acc = correct_preds / len(test_loader.dataset)
print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_acc * 100):.2f}%")
