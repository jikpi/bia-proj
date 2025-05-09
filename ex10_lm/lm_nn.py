import torch
from torch import nn


# klasicka ANN s relu a dropout proti overfitting
class LogisticMapNN(nn.Module):
    def __init__(self, input_dim):
        super(LogisticMapNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ModelCheckpoint:
    def __init__(self, filepath=None, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.best_state_dict = None

    def __call__(self, metrics, model):
        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
            self.best_state_dict = model.state_dict().copy()
            self._save_model(model)
        elif (self.mode == 'min' and current_score < self.best_score) or \
                (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.best_state_dict = model.state_dict().copy()
            self._save_model(model)

    def _save_model(self, model):
        if self.filepath:
            torch.save(model.state_dict(), self.filepath)

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mae += torch.mean(torch.abs(outputs - targets)).item()

    return running_loss / len(dataloader), running_mae / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            running_mae += torch.mean(torch.abs(outputs - targets)).item()

    return running_loss / len(dataloader), running_mae / len(dataloader)
