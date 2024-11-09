import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUMBER_OF_OUTPUTS = 1
NUMBER_OF_CHOICES = 2
BATCH_SIZE = 40
IMAGE_DIMENSIONS = 64
NEURONS = 2674
DEVICE = "cpu"  # Use "cuda" if you have CUDA and want to use the GPU

transform_bad = transforms.Compose([
    transforms.Resize((IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3598, 0.4181, 0.4291], std=[0.2876, 0.2580, 0.2811]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-180, 180)),
    transforms.Grayscale()
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4282, 0.4245, 0.3755], std=[0.3002, 0.2806, 0.2890]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-180, 180)),
    transforms.Grayscale()
])

bad_dataset = datasets.ImageFolder(root="Bad_data", transform=transform_bad)
test_dataset = datasets.ImageFolder(root="Test_Dataset", transform=transform_test)

bad_dataloader = DataLoader(bad_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class AI1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_DIMENSIONS*IMAGE_DIMENSIONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NUMBER_OF_CHOICES)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = AI1().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

X = torch.rand(1, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, device=DEVICE)
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
    train(bad_dataloader, model, loss_fn, optimizer)
    trainEnd = time.time()
    print(f"Total train time: {trainEnd - trainStart}")

    testStart = time.time()
    test(test_dataloader, model, loss_fn)
    testEnd = time.time()
    print(f"Total test time: {testEnd - testStart}")

print("Finished!")
