import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from colorama import Fore, Back, Style
from tqdm import tqdm
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.1)

train_loss_list = []
train_accuracy_list = []

num_epochs = 30

fin = num_epochs * len(train_loader)
progress = 0

with tqdm(train_loader, desc=f'Training : ', total=fin) as train_bar:
    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            progress += 16
            if (batch_idx + 1) % 100 == 0:
                # print(Fore.GREEN + 'Epoch' + Fore.WHITE + f': [{epoch + 1}/{num_epochs}] * ' + Fore.YELLOW + 'Step' + Fore.WHITE + f': [{batch_idx + 1}/{len(train_loader)}] * ' + Fore.RED + 'Loss' + Fore.WHITE + f': {loss.item():.4f}')
                train_loss_list.append(train_loss / len(train_loader))
                train_accuracy = correct / total
                train_accuracy_list.append(train_accuracy)
            train_bar.set_postfix({'Loss: ': train_loss / (batch_idx + 1)})
            train_bar.update()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(Fore.LIGHTMAGENTA_EX + f'Test Accuracy' + f': {accuracy * 100:.2f}%')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Training Loss')
# plt.axis((1, num_epochs, 0, 1))
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy_list)+1), train_accuracy_list, label='Training Accuracy')
# plt.axis((1, num_epochs, 0, 1))
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()