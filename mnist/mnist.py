import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from colorama import Fore, Back, Style

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

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

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (batch_idx + 1) % 100 == 0:
            print(Fore.GREEN + 'Epoch' + Fore.WHITE + f': [{epoch + 1}/{num_epochs}] / ' + Fore.YELLOW + 'Step' + Fore.WHITE + f': [{batch_idx + 1}/{len(train_loader)}] / ' + Fore.RED + 'Loss' + Fore.WHITE + f': {loss.item():.4f}')

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