from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST as SlowMNIST
from torch.optim.lr_scheduler import StepLR
import wandb

epochs = 4
batch_size = 50
learning_rate = 1

test_batch_size = 1000

wandb.init(
    dir=Path(__file__).parent.parent / "wandb", project="polynet-preserved-concept"
)

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"
device = torch.device(device_type)


class MNIST(SlowMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class PolyNet(nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model: PolyNet, device: torch.device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            wandb.log({"train/loss": round(loss.item(), 6), "train/epoch": epoch})


def test(model: PolyNet, device: torch.device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)\n"
    )
    wandb.log(
        {
            "test/average_loss": round(test_loss, 4),
            "test/accuracy": 100 * correct / len(test_loader.dataset),
        }
    )


def main():
    datasets_path = Path(__file__).parent.parent / "datasets"
    train_dataset = MNIST(datasets_path, train=True, download=True)
    test_dataset = MNIST(datasets_path, train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10000, shuffle=False, num_workers=0
    )

    model = PolyNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, epochs + 1):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)
        scheduler.step()

    save_file = Path(Path(__file__).parent.parent / "mnist_cnn_fast.pth").as_posix()
    torch.save(model.state_dict(), save_file)
    wandb.save(save_file)


if __name__ == "__main__":
    main()
