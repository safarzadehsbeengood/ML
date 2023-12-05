import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from colorama import Fore
from tqdm import tqdm
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.EMNIST(root='./data', train=True, split='balanced', download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root='./data', train=False, split='balanced', download=True, transform=transform)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def objective(params):
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']

    model = NN(dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, targets in train_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))
    pbar.update()
    return train_loss_list[-1]

# Define the hyperparameter search space
space = {
    'num_epochs': hp.choice('num_epochs', [50, 100, 150]),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.8),
}

# Choose the optimization algorithm (Tree of Parzen Estimators - TPE)
algorithm = tpe.suggest
# Run the optimization process
with tqdm(desc='Optimizing...', total = 10) as pbar:
    best = fmin(fn=objective,  # Objective function to minimize
                space=space,     # Search space for hyperparameters
                algo=algorithm,  # Optimization algorithm
                max_evals=10)     # Number of evaluations

# Print the best set of hyperparameters found
print("Best hyperparameters:", best)

