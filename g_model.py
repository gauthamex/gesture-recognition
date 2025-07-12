import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----- STEP 1: Load Dataset -----
data_dir = "leapGestRecog"
img_size = 64
X, y = [], []

sample_user_folder = os.path.join(data_dir, '00')
gestures = sorted(os.listdir(sample_user_folder))
label_map = {gesture: idx for idx, gesture in enumerate(gestures)}

print(f"Gesture Labels: {label_map}")

for user in os.listdir(data_dir):
    user_path = os.path.join(data_dir, user)
    for gesture in os.listdir(user_path):
        gesture_path = os.path.join(user_path, gesture)
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_map[gesture])

X = np.array(X).reshape(-1, 1, img_size, img_size).astype(np.float32) / 255.0
y = np.array(y).astype(np.int64)

# ----- STEP 2: Split Dataset -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = GestureDataset(X_train, y_train)
test_ds = GestureDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ----- STEP 3: Build Model -----
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=len(label_map)).to(device)

# ----- STEP 4: Train Model -----
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training started...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
print("Training finished.")

# ----- STEP 5: Evaluate -----
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc * 100:.2f}%")

# ----- STEP 6: Save Model -----
torch.save(model.state_dict(), "hand_gesture_model.pth")
print("Model saved as hand_gesture_model.pth")
