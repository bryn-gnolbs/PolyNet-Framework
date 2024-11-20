import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import time
from pathlib import Path

# Initialize Weights & Biases (wandb)
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
print(f"using: {device_type}")

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 50
gradient_clip_value = 3.0

# Data loading and preprocessing
datasets_path = Path(__file__).parent.parent / "datasets"
transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root=datasets_path, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    root=datasets_path, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Define adaptive spline activation
class AdaptiveSplineActivation(nn.Module):
    def __init__(self):
        super(AdaptiveSplineActivation, self).__init__()
        self.coefficients = nn.Parameter(
            torch.randn(3)
        )  # Learnable coefficients for the spline

    def forward(self, x):
        # Spline function with learnable coefficients
        return (
            self.coefficients[0] * x**2
            + self.coefficients[1] * x
            + self.coefficients[2]
        )


# Define dynamic temporal processing layer
class TemporalFeedbackLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalFeedbackLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.fc = nn.Linear(input_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_state = None

    def forward(self, x):
        if self.hidden_state is None or self.hidden_state.size(0) != x.size(0):
            self.hidden_state = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        combined = torch.cat((x, self.hidden_state.detach()), dim=1)
        self.hidden_state = self.relu(self.fc(combined))
        return self.hidden_state


# Define the preserved PolyNet model
class PolyNet(nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Temporal feedback layer
        self.temporal_layer = TemporalFeedbackLayer(
            input_size=32 * 8 * 8, hidden_size=256
        )

        # Fully connected layers with adaptive spline activation
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.spline1 = AdaptiveSplineActivation()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.spline2 = AdaptiveSplineActivation()
        self.fc3 = nn.Linear(64, 10)

        # Dropout to prevent overfitting
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, x):
        # Convolutional layers for feature extraction
        x = self.relu(self.pool(self.conv1(x)))
        x = self.dropout1(self.relu(self.pool(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)

        # Temporal feedback layer
        x = self.temporal_layer(x)

        # Fully connected layers with adaptive spline activation
        x = self.bn1(self.spline1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.bn2(self.spline2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
polynet_model = PolyNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(polynet_model.parameters(), lr=learning_rate, momentum=0.95)

# Learning rate scheduler (using warm-up followed by Cosine Annealing)
warmup_epochs = 7
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        ),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs),
    ],
    milestones=[warmup_epochs],
)


# Training loop with gradient clipping and energy tracking
def train_model(model, optimizer, scheduler, model_name):
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()

            # Print statistics and log to wandb
            running_loss += loss.item()
            if i % 100 == 99:
                wandb.log(
                    {f"{model_name}_loss": running_loss / 100, "epoch": epoch + 1}
                )
                print(
                    f"{model_name} - Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

        # Step the learning rate scheduler
        scheduler.step()

    # Calculate and log time and energy consumption
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training time for {model_name}: {total_time:.2f} seconds")
    wandb.log(
        {
            f"{model_name}_training_time": total_time,
        }
    )

    print(f"Training completed for {model_name}")


# Train the model
train_model(polynet_model, optimizer, scheduler, "PolyNet")


# Test the model
def test_model(model, model_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set for {model_name}: {accuracy:.2f}%")
    wandb.log({f"{model_name}_test_accuracy": accuracy})


# Test the model
test_model(polynet_model, "PolyNet")

# Save the
save_file = Path(
    Path(__file__).parent.parent / "polynet_preserved_concept.pth"
).as_posix()
torch.save(polynet_model.state_dict(), save_file)
wandb.save(save_file)
