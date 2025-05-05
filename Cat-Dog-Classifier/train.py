import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from network import ClassifierNetwork

TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 20

# Load saved numpy arrays X and y
dataset = np.load("Cat-Dog-Classifier/Data/cat_dog_dataset.npz")
X = dataset["X"] # (num_images, img_size, img_size, num_channels)
y = dataset["y"] # (num_images,)

# Split into train and test sets
train_size = int(X.shape[0] * TRAIN_SPLIT)
train_size = train_size - (train_size % BATCH_SIZE)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Train and test datasets
# [num_images, img_size, img_size, num_channels] -> [num_images, num_channels, img_size, img_size] and normalize to [0, 1]
train_dataset = TensorDataset(torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_train).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y_test).float())

# Train and test dataloaders
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Decay the learning rate by 25% every epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"----- Epoch {epoch + 1}/{NUM_EPOCHS} -----")

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
        loss = F.binary_cross_entropy(y_pred, y_batch)

        total_train_loss += loss.item()
        correct_preds += ((y_pred >= 0.5) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training loss and accuracy
    train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    train_acc = correct_preds / len(train_loader.dataset)
    train_acc_history.append(train_acc)

    print(f"Train loss: {train_loss:.4f} - Train accuracy: {(train_acc * 100):.2f}%")

    # Set the model to evaluation mode
    model.eval()

    total_test_loss = 0
    correct_preds = 0
    
    for X_batch, y_batch in test_loader:
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        with torch.no_grad():
            y_pred = model(X_batch).squeeze() # [batch_size, 1] -> [batch_size]

        # Calculate the loss and number of correct predictions
        total_test_loss += F.binary_cross_entropy(y_pred, y_batch).item()
        correct_preds += ((y_pred >= 0.5) == y_batch).sum().item()

    # Calculate test loss and accuracy
    test_loss = total_test_loss / len(test_loader)
    test_loss_history.append(test_loss)

    test_acc = correct_preds / len(test_loader.dataset)
    test_acc_history.append(test_acc)

    print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_acc * 100):.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "Cat-Dog-Classifier/Training/Model_Checkpoint.pth")

        print("----- Saved Model Checkpoint -----")

    scheduler.step()

# Plot training and test loss and accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
epochs = [i for i in range(1, NUM_EPOCHS + 1)]

# Loss vs Epoch graph
axes[0].plot(epochs, train_loss_history, label="Train Loss")
axes[0].plot(epochs, test_loss_history, label="Test Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy vs Epoch graph
axes[1].plot(epochs, train_acc_history, label="Train Accuracy")
axes[1].plot(epochs, test_acc_history, label="Test Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.savefig("Cat-Dog-Classifier/Training/Training_Graph.png")
plt.show()