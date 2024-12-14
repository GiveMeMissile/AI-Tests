import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

NUMBER_OF_CHOICES = 10
BATCH_SIZE = 100
IMAGE_DIMENSIONS = 28
NEURONS = 512
DEVICE = "cpu"  # Set to "cuda" if using GPU

class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = int(self.data.iloc[idx, 0])  # Ensure label is an integer
        image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape((28, 28))  # Convert to 28x28 format

        if self.transform:
            # Use transform directly on the image as a 2D array
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = MNISTDataset(csv_file="Good_dataset/mnist_train.csv", transform=transform)
test_dataset = MNISTDataset(csv_file="Good_dataset/mnist_test.csv", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class AI2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_DIMENSIONS * IMAGE_DIMENSIONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NUMBER_OF_CHOICES)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = AI2().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Test model's output with a dummy image (single grayscale channel)
X = torch.rand(1, 1, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, device=DEVICE)  # Adjusted shape
logits = model(X)
predict_probability = nn.Softmax(dim=1)(logits)
y_predict = predict_probability.argmax(1)
print(f"Predicted class: {y_predict}")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Average Loss: {test_loss:>8f}")


def getEpoch():
    while True:
        try:
            epochs = int(input("How many epochs do you desire?: "))
            if epochs > 0:
                break
            else:
                print("Please input a number greater than 0.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    return epochs


epochs = getEpoch()
for t in range(epochs):
    print(f"Epoch {t + 1}\n------------------------")
    trainStart = time.time()
    train(train_dataloader, model, loss_fn, optimizer)
    trainEnd = time.time()
    print(f"Total train time: {trainEnd - trainStart}")

    testStart = time.time()
    test(test_dataloader, model, loss_fn)
    testEnd = time.time()
    print(f"Total test time: {testEnd - testStart}")

print("Finished!")
