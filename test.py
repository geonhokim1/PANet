import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

from dataloader import Dataset
from model import PANet


### Device ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Dataloader ###
X_train_np = np.load("./datasets/X_train.npy")
scaler = StandardScaler().fit(X_train_np.reshape(-1, 2))

test_set = Dataset("./datasets/X_test.npy", "./datasets/y_test.npy", scaler=scaler, augment=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


### Model ###
model = PANet().to(device)
model.load_state_dict(torch.load('./PANet.pt', weights_only=False))
model.eval()


### Test ###
test_correct = 0
test_num = 0
with torch.no_grad():
    for xb, yb in test_loader:
        x, y_true = xb.to(device), yb.to(device)

        y_pred = model(x)

        y_true_idx = torch.argmax(y_true, dim=1).long()
        y_pred_idx = torch.argmax(y_pred, dim=1).long()
        
        r = (y_pred_idx == y_true_idx).sum().item()

        test_correct += r
        test_num += y_true.size(0)

    print(f'Accuracy: {100 * test_correct/test_num}')

    