import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DoubleInverseSEM(nn.Module):
    def __init__(self):
        super(DoubleInverseSEM, self).__init__()

    def forward(self, x):
        numerator = torch.e**-x
        e_val = torch.exp(torch.tensor(1.0, device=x.device))
        exponents = torch.arange(1, e_val + 1, device=x.device)  # type: ignore
        denominator = torch.sum(torch.exp(-x.unsqueeze(1) * exponents), dim=1)
        return numerator / denominator


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.activation = DoubleInverseSEM()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
        )


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


def main():
    train_loader, test_loader = load_data()
    model = TestNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    print("Evaluating the model...")
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
