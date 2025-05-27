import os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import PANet
from dataloader import Dataset
from visual_tools import plot_graph

### Fix Seed ###
def set_seed(seed=304):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(304) # our course num


### Device ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Dataloader ###
X_train_np = np.load("./datasets/X_train.npy")
scaler = StandardScaler().fit(X_train_np.reshape(-1, 2))

train_set = Dataset("./datasets/X_train.npy", "./datasets/y_train.npy", scaler=scaler, augment=True)
valid_set = Dataset("./datasets/X_valid.npy", "./datasets/y_valid.npy", scaler=scaler, augment=False)

batch_size=32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


### Model ###
model = PANet().to(device)


### Loss ###
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = 0
train_accs = []
train_losses = []
val_accs =[]
val_losses = []


### Train setting ###
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.7,
    patience=10,
    verbose=True
)

epochs = 200


### Train ###
for epoch in range(1, epochs + 1):
    model.train()
    train_acc = 0.0
    train_correct = 0
    train_loss = 0.0
    train_num = 0

    ### Train ###
    for xb, yb in train_loader:
        x, y_true = xb.to(device), yb.to(device)

        y_pred = model(x)

        y_true_idx = torch.argmax(y_true, dim=1).long()
        y_pred_idx = torch.argmax(y_pred, dim=1).long()

        train_correct += (y_pred_idx == y_true_idx).sum().item()

        loss = criterion(y_pred, y_true_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)
    train_acc = 100 * train_correct / train_num
    train_loss = train_loss / train_num
    train_accs.append(train_acc)
    train_losses.append(train_loss)

    print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f}%  Train Loss: {train_loss:.4f}")

    ### Valid ###
    model.eval()
    val_acc = 0.0
    val_correct = 0
    val_loss = 0.0
    val_num = 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            x, y_true = xb.to(device), yb.to(device)

            y_pred = model(x)

            y_true_idx = torch.argmax(y_true, dim=1).long()
            y_pred_idx = torch.argmax(y_pred, dim=1).long()

            val_correct += (y_pred_idx == y_true_idx).sum().item()

            loss = criterion(y_pred, y_true_idx)
            val_loss += loss.item() * x.size(0)
            val_num += y_true.size(0)

        val_acc = 100 * val_correct / val_num
        val_loss = val_loss / val_num
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        print(f"           Val Acc: {val_acc:.4f}%    Val Loss: {val_loss:.4f}")
        scheduler.step(val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'./PANet.pt')
            print('==> best model saved | Name: PANet.pt')

    ### Visualization ###
    if epoch % 50 == 0:
        plot_graph(train_accs, val_accs, epoch, "Accuracy")
        plot_graph(train_losses, val_losses, epoch, "Loss")
