import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# The higher magnitude bins the more augmentations, the max is 31
MAGNITUDE_BINS = 31
TRAIN_ROOT = "Bad_data"
# Test dataset will change when I get a better test dataset.
TEST_ROOT = "Test_Dataset"

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


class CNNModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, num_hidden_layers=1):
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
            nn.Linear(in_features=hidden_features*int(64/(num_hidden_layers**2)), out_features=output_features)
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


rand = torch.randn(size=(3, 64, 64), dtype=torch.float32)
model = CNNModel(input_features=3, hidden_features=16, output_features=3, num_hidden_layers=2)
model.eval()
with torch.inference_mode():
    dummy_prediction = model.forward(rand, show_hidden=True)
    print(f"Worthless prediction: {dummy_prediction}")


