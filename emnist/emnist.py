import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from colorama import Fore, Back, Style
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: " + str(device))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.EMNIST(root='./data', train = True, split='balanced', download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root='./data', train = False, split='balanced', download=True, transform=transform)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# print('len(train_loader): ' + str(len(train_loader)))

class NN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 47)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x
    

num_epochs = 10
learning_rate = 0.001
dropout_rate = 0.5

model = NN(dropout_rate=0).to(device)
criterion = nn.CrossEntropyLoss().to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

train_loss_list = []
train_accuracy_list = []

its = num_epochs * len(train_loader)

with tqdm(train_loader, desc=f'Training... ', total=its) as train_bar:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data).to(device)
            loss = criterion(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            train_bar.update()
            train_bar.set_postfix({'Loss: ': train_loss / (batch_idx + 1)})

            # if (batch_idx + 1) % 50 == 0:
                # print(Fore.GREEN + 'Epoch' + Fore.WHITE + f': [{epoch + 1}/{num_epochs}] * ' + Fore.YELLOW + 'Step' + Fore.WHITE + f': [{batch_idx + 1}/{len(train_loader)}] * ' + Fore.RED + 'Loss' + Fore.WHITE + f': {loss.item():.4f}')
        train_loss_list.append(train_loss / len(train_loader))
        train_accuracy = correct / total
        train_accuracy_list.append(train_accuracy)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(Fore.GREEN + f'Test Accuracy' + Fore.WHITE + f': {accuracy * 100:.2f}%')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
# plt.axis((1, num_epochs, 0, 1))
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
y=train_accuracy_list
plt.plot(range(1, num_epochs+1), train_accuracy_list, label='Training Accuracy')
# plt.axis((1, num_epochs, 0, 1))
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()