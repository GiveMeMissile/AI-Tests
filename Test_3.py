import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from pathlib import Path

# The higher magnitude bins the more augmentations, the max is 31
MAGNITUDE_BINS = 31
TRAIN_ROOT = "Bad_data"
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 16
# Test dataset will change when I get a better test dataset.
TEST_ROOT = "Placeholder"


train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=MAGNITUDE_BINS),  # Data augmentation, can be removed if desired
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=train_transform)
test_data = datasets.ImageFolder(root=TEST_ROOT, transform=test_transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, input_features, hidden_features, num_hidden_layers=1, output_features=1):
        super().__init__()
        self.input_layer = nn.Conv2d(in_channels=input_features, out_channels=hidden_features,
                                       kernel_size=3, stride=1, padding=1)
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*16*16, out_features=output_features)
        )
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x, show_hidden=False):
        x = self.input_layer(x)
        # this is for easy usage of a different amount of hidden layers. Makes is very good for testing
        for i in range(self.num_hidden_layers):
            x = self.hidden_layer(x)
            if show_hidden:
                print(f"Hidden layer {i+1}")
                print(x.shape)
        x = self.output_layer(x)
        return x


rand = torch.randn(size=(BATCH_SIZE, 3, 64, 64), dtype=torch.float32)
model = CNNModel(input_features=3, hidden_features=16, num_hidden_layers=2)
'''
model.eval()
with torch.inference_mode():
    dummy_prediction = model.forward(rand, show_hidden=True)
    print(f"Worthless prediction: {dummy_prediction}")
'''
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters())


def train(model, loss_fn, optimizer, dataloader):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in enumerate(dataloader):
        y_predictions = model(X).squeeze(dim=1)
        loss = loss_fn(y_predictions.type(torch.float32), y.type(torch.float32))
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_prediction_class = (torch.sigmoid(y_predictions) > 0.5).int()
        train_acc += (y_prediction_class == y).sum().item() / len(y)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc*100


def test(model, loss_fn, dataloader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            test_predictions = model(X).squeeze(dim=1)
            loss = loss_fn(test_predictions.type(torch.float32), y.type(torch.float32))
            test_loss += loss
            test_prediction_class = (torch.sigmoid(test_predictions) > 0.5).int()
            test_acc += (test_prediction_class == y).sum().item() / len(y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc*100


def get_epoch():
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


def create_and_display_dataframe(dictionary):
    dataframe = pd.DataFrame(dictionary)
    print(dataframe)
    return dataframe


def save_model(model):
    save = input("Do you want to save this model?: ")
    if not (save == "yes" or save == "Yes"):
        print("This model shall not be saved.")
        return 0
    model_path = Path(input("What file do you want to save your model in?: "))
    model_path.mkdir(parents=True, exist_ok=True)
    model_name = input("What do you want your model dict's name to be?: ")+".pth"
    model_save_path = model_path/model_name
    print("Now downloading the model.....")
    torch.save(obj=model.state_dict(), f=model_save_path)
    print("Model successfully saved! YIPPEE")


if __name__ == "__main__":
    epochs = get_epoch()
    results = {"Epoch": [], "Train_loss": [], "Train_accuracy": [], "Test_loss": [], "Test_accuracy": [], "Time": []}
    print("We shall now begin the training process.")

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train(model, loss_fn, optimizer, train_dataloader)
        test_loss, test_acc = test(model, loss_fn, test_dataloader)
        end = time.time()
        results["Epoch"].append(epoch + 1)
        results["Train_loss"].append(train_loss.detach().item())
        results["Train_accuracy"].append(train_acc)
        results["Test_loss"].append(test_loss.detach().item())
        results["Test_accuracy"].append(test_acc)
        results["Time"].append(end-start)
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.3f}%\n"
            f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.3f}%, Time: {end-start:.3f}s.\n")

    results_df = create_and_display_dataframe(dictionary=results)
    save_model(model)
