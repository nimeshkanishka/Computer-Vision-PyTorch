import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from custom_network import CustomClassifierNetwork
from pretrained_networks import EfficientNet_B0
from config import IMAGE_SIZE, RGB, NETWORK, BATCH_SIZE, NUM_EPOCHS

# Load training dataset
dataset_train = np.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Training-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz")
X_train = dataset_train["X"] # (num_images, img_size, img_size, num_channels)
y_train = dataset_train["y"] # (num_images,)

# Load validation dataset
dataset_valid = np.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Validation-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz")
X_val = dataset_valid["X"] # (num_images, img_size, img_size, num_channels)
y_val = dataset_valid["y"] # (num_images,)

# Load test dataset
dataset_test = np.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Data/Test-Dataset-{IMAGE_SIZE}x{IMAGE_SIZE}-{'RGB' if RGB else 'Gray'}.npz")
X_test = dataset_test["X"] # (num_images, img_size, img_size, num_channels)
y_test = dataset_test["y"] # (num_images,)

# Training, validation and test tensor datasets
# X: Shape: [num_images, img_size, img_size, num_channels] -> [num_images, num_channels, img_size, img_size]
#    Dtype: float32 (Normalized to [0, 1])
# y: Shape: [num_images]
#    Dtype: float32
train_dataset = TensorDataset(torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_train).float())
val_dataset = TensorDataset(torch.from_numpy(X_val).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_val).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_test).float())

# Training, validation and test dataloaders
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match NETWORK:
    case "Custom":
        model = CustomClassifierNetwork().to(device)
    case "EfficientNet-B0":
        model = EfficientNet_B0().to(device)
    case _:
        raise ValueError(f"Unsupported network architecture: '{NETWORK}'. Choose from 'Custom' or 'EfficientNet-B0'.")
    
# AdamW optimizer applies weight decay (L2 regularization) and adapts learning rates per parameter
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Increase the learning rate from 4e-5 to 1e-3 and then decrease it to 4e-9 over training steps using cosine annealing
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

best_val_loss = 1000000
best_val_acc = 0

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
        y_pred = model(X_batch).squeeze() # [batch_size, 1] -> [batch_size]

        # Calculate the loss
        loss = F.binary_cross_entropy_with_logits(y_pred, y_batch)

        total_train_loss += loss.item()
        correct_preds += ((torch.sigmoid(y_pred) >= 0.5) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the optimizer's learning rate after every batch
        scheduler.step()

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
    
    for X_batch, y_batch in val_loader:
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        with torch.no_grad():
            y_pred = model(X_batch).squeeze() # [batch_size, 1] -> [batch_size]

        # Calculate the loss and number of correct predictions
        total_val_loss += F.binary_cross_entropy_with_logits(y_pred, y_batch).item()
        correct_preds += ((torch.sigmoid(y_pred) >= 0.5) == y_batch).sum().item()

    # Calculate validation loss and accuracy
    val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(val_loss)

    val_acc = correct_preds / len(val_loader.dataset)
    val_acc_history.append(val_acc)

    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {(val_acc * 100):.2f}%")

    # Save model checkpoint if we have completed more than 10 epochs AND validation accuracy has increased or
    # validation loss has decreased with validation accuracy remaining the same
    if (epoch > 9) and ((val_acc > best_val_acc) or ((val_acc == best_val_acc) and (val_loss <= best_val_loss))):
        best_val_loss = val_loss
        best_val_acc = val_acc
        
        # Save model checkpoint
        torch.save(model.state_dict(),
                   f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Training/Model-Checkpoint-{NETWORK}.pth")
        
        print(f"----- Model checkpoint saved to 'Normal-vs-Pneumonia-Chest-X-Ray-Classification/Training/Model-Checkpoint-{NETWORK}.pth' -----")

# Plot training and validation loss and accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
epochs = [i for i in range(1, NUM_EPOCHS + 1)]

# Training/Validation Loss vs Epoch graph
axes[0].plot(epochs, train_loss_history, label="Train Loss")
axes[0].plot(epochs, val_loss_history, label="Validation Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Training/Validation Accuracy vs Epoch graph
axes[1].plot(epochs, train_acc_history, label="Train Accuracy")
axes[1].plot(epochs, val_acc_history, label="Validation Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.savefig(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Training/Training-Graph-{NETWORK}.png")
plt.show()

"""
Testing
"""
# Load the saved weights from the best epoch and set the model to evaluation mode
model.load_state_dict(torch.load(f"Normal-vs-Pneumonia-Chest-X-Ray-Classification/Training/Model-Checkpoint-{NETWORK}.pth"))
model.eval()

total_test_loss = 0
correct_preds = 0

for X_batch, y_batch in test_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        y_pred = model(X_batch).squeeze() # [batch_size, 1] -> [batch_size]

    total_test_loss += F.binary_cross_entropy_with_logits(y_pred, y_batch).item()
    correct_preds += ((torch.sigmoid(y_pred) >= 0.5) == y_batch).sum().item()

test_loss = total_test_loss / len(test_loader)
test_acc = correct_preds / len(test_loader.dataset)

print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_acc * 100):.2f}%")