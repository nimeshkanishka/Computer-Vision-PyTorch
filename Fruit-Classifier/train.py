import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from network import ClassifierNetwork
from config import CATEGORIES, BATCH_SIZE, NUM_EPOCHS

# Load training dataset
dataset_train = np.load("Fruit-Classifier/Data/Training-Dataset.npz")
X_train = dataset_train["X"] # (num_images, img_size, img_size, num_channels)
y_train = dataset_train["y"] # (num_images,)

# Load validation dataset
dataset_valid = np.load("Fruit-Classifier/Data/Validation-Dataset.npz")
X_valid = dataset_valid["X"] # (num_images, img_size, img_size, num_channels)
y_valid = dataset_valid["y"] # (num_images,)

# Load test dataset
dataset_test = np.load("Fruit-Classifier/Data/Test-Dataset.npz")
X_test = dataset_test["X"] # (num_images, img_size, img_size, num_channels)
y_test = dataset_test["y"] # (num_images,)

# Training, validation and test tensor datasets
# X: Shape: [num_images, img_size, img_size, num_channels] -> [num_images, num_channels, img_size, img_size]
#    Dtype: float32 (Normalized to [0, 1])
# y: Shape: [num_images]
#    Dtype: int64
train_dataset = TensorDataset(torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_train))
valid_dataset = TensorDataset(torch.from_numpy(X_valid).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_valid))
test_dataset = TensorDataset(torch.from_numpy(X_test).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_test))

# Training, validation and test dataloaders
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierNetwork(num_classes=len(CATEGORIES)).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
# Decay the learning rate using cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(NUM_EPOCHS):
    print(f"----- Epoch {epoch + 1}/{NUM_EPOCHS} -----")

    """
    Training
    """
    # Set the model to training mode
    model.train()

    total_train_loss = 0
    correct_preds = 0

    for X_batch, y_batch in tqdm(train_loader):
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        y_pred = model(X_batch) # [batch_size, num_classes]

        # Calculate the loss
        loss = F.cross_entropy(y_pred, y_batch)

        total_train_loss += loss.item()
        correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training loss and accuracy
    train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    train_acc = correct_preds / len(train_loader.dataset)
    train_acc_history.append(train_acc)

    print(f"Train loss: {train_loss:.4f} - Train accuracy: {(train_acc * 100):.2f}%")

    """
    Validation
    """
    # Set the model to evaluation mode
    model.eval()

    total_val_loss = 0
    correct_preds = 0
    
    for X_batch, y_batch in valid_loader:
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        with torch.no_grad():
            y_pred = model(X_batch) # [batch_size, num_classes]

        # Calculate the loss and number of correct predictions
        total_val_loss += F.cross_entropy(y_pred, y_batch).item()
        correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

    # Calculate validation loss and accuracy
    val_loss = total_val_loss / len(valid_loader)
    val_loss_history.append(val_loss)

    val_acc = correct_preds / len(valid_loader.dataset)
    val_acc_history.append(val_acc)

    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {(val_acc * 100):.2f}%")

    scheduler.step()

# Plot training and test loss and accuracy
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

plt.savefig("Fruit-Classifier/Training/Training_Graph.png")
plt.show()

"""
Testing
"""
model.eval()

total_test_loss = 0
correct_preds = 0

for X_batch, y_batch in test_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        y_pred = model(X_batch) # [batch_size, num_classes]

    total_test_loss += F.cross_entropy(y_pred, y_batch).item()
    correct_preds += (y_pred.argmax(dim=1) == y_batch).sum().item()

test_loss = total_test_loss / len(test_loader)
test_acc = correct_preds / len(test_loader.dataset)

print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_acc * 100):.2f}%")

# Save model weights
torch.save(model.state_dict(), "Fruit-Classifier/Training/Model_Checkpoint.pth")