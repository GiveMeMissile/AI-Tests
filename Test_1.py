import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

NUMBER_OF_OUTPUTS = 1
NUMBER_OF_CHOICES = 2
TRAIN_BATCH_SIZE = 20
IMAGE_DIMENSIONS = 64
NEURONS = 2674
DEVICE = "cpu" #This will be used on CPU for me. You can change it to CUDA if you have cuda downloaded.

transform = transforms.Compose([transforms.Resize((IMAGE_DIMENSIONS,IMAGE_DIMENSIONS)) ,transforms.ToTensor(),
    transforms.Normalize(mean=[0.3598, 0.4181, 0.4291], std=[0.2876, 0.2580, 0.2811]),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-180, 180)), ])

bad_dataset = datasets.ImageFolder(root="Bad_data", transform=transform)
bad_dataloader = DataLoader(bad_dataset, batch_size= TRAIN_BATCH_SIZE, shuffle=True)

class AI1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_DIMENSIONS*IMAGE_DIMENSIONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NEURONS),
            nn.ReLU(),
            nn.Linear(NEURONS, NUMBER_OF_CHOICES))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = AI1().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

X = torch.rand(NUMBER_OF_OUTPUTS, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, device=DEVICE)
logits = model(X)
predict_probability = nn.Softmax(dim=NUMBER_OF_OUTPUTS)(logits)
y_predict = predict_probability.argmax(NUMBER_OF_OUTPUTS)
print(f"Predicted class: {y_predict}")

def train(bad_dataloader, model, loss_fn, optimizer):
    size = len(bad_dataloader.bad_dataset)
    model.train()
    for batch, (X,y) in enumerate(bad_dataloader):
        X = X.to.DEVICE
        y = y.to.DEVICE
        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backwards()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = (batch+1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
