import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from random import shuffle
from time import time
from pandas import DataFrame
from pathlib import Path

# CONSTANTS!!!
TRAIN_DATA = "copycat-project/laion2b6plus_fish"
TEST_DATA = "Bad_data"
BATCH_SIZE = 32
IMAGE_DIMENSIONS = 64
COLOR_CHANNELS = 3
LEARNING_RATE = 0.001
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS)),
    transforms.ToTensor()
])


class DataPreparer:
    def __init__(self, data):
        # Stores a dataset to change
        self.dataset = data

    def shuffle_data(self, dataset=None):
        # As the title suggests this function shuffles the data
        if dataset is None:
            dataset = self.dataset
        merged_data = list(zip(dataset["image"], dataset["category"]))
        shuffle(merged_data)
        dataset["image"], dataset["category"] = zip(*merged_data)
        self.dataset["image"] = dataset["image"]
        self.dataset["category"] = dataset["category"]
        return dataset

    def batch_data(self, dataset=None):
        # This function batches the data in order to train a neural network.
        if dataset is None:
            dataset = self.dataset
        dataset = self.tensorize_category(dataset)
        batch = 1
        images = []
        categories = []
        X_list = []
        y_list = []
        for i, (image, category) in enumerate(zip(dataset["image"], dataset["category"])):
            images.append(image)
            categories.append(category)
            if i+1 == BATCH_SIZE*batch or len(dataset["image"]) == i+1:
                batch += 1
                X = torch.stack(images, dim=0)
                y = torch.stack(categories, dim=0)
                images.clear()
                categories.clear()
                X_list.append(X)
                y_list.append(y)
        dataset = {"X": X_list, "y": y_list}
        self.dataset = dataset
        return dataset

    def tensorize_category(self, dataset=None):
        # This turns the category from strings to torch tensors that can be used in a Neural Network
        if dataset is None:
            dataset = self.dataset
        categories = []
        for category in dataset["category"]:
            if category == "fish":
                y = torch.tensor([0])
            else:
                y = torch.tensor([1])
            categories.append(y)

        dataset["category"] = categories
        self.dataset = dataset
        return dataset

    def transform(self, examples):
        # This transforms the huggingface dataset using the TRANSFORMER
        examples["pixel_values"] = [TRANSFORM(image) for image in examples["image"]]
        return examples

    def synthesize_data(self, dataset=None):
        # This function creates images using torch.rand that mimic the non_fish category.
        # This function returns a dictionary containing both the original images, random images,
        # and the categories for both.
        if dataset is None:
            dataset = self.dataset
        print("\nSynthesizing data...")
        replicated_data = []
        replicated_category = []

        for i in range(len(dataset)):
            X = torch.rand(size=(COLOR_CHANNELS, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS))
            replicated_data.append(X)
            replicated_category.append("non_fish")

        dataset.set_transform(self.transform)
        dataset.to_dict()

        pixel_values = []

        for values in dataset:
            pixel_values.append(values["pixel_values"])

        category = []

        for categories in dataset:
            category.append(categories["category"])

        data = {"image": pixel_values + replicated_data,
                "category": category + replicated_category}

        self.dataset = data

        print("Data successfully synthesized")
        return data

    def prepare_data(self, dataset=None):
        # This function takes a dataset and prepares it using the other functions in the class.
        if dataset is None:
            dataset = self.dataset
        dataset = self.synthesize_data(dataset)
        dataset = self.shuffle_data(dataset)
        dataset = self.batch_data(dataset)
        return dataset


class AdaptiveCNNModel(nn.Module):
    # I am using a model that I used in Test 2
    # This cnn is made to be easily altered which makes it good for testing.
    # And this model is capable of reaching a high accuracy as well.

    def __init__(self, input_features, hidden_features, hidden_layers=1, output_features=1):
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
        self.output_fixer = lambda length, layers: int((length*((1/2)**layers))**2)

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*self.output_fixer(IMAGE_DIMENSIONS, hidden_layers),
                      out_features=output_features)
        )
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = self.input_layer(x)
        # This model can cycle through the hidden layer multiple times!!! (WOW)
        for i in range(self.hidden_layers):
            x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def get_huggingface_data():
    # This function gets the required dataset from hugging face and returns it.
    print("Getting huggingface dataset...")
    datasets = load_dataset(TRAIN_DATA, split="train")
    datasets = datasets.cast_column("image", Image(mode="RGB"))
    datasets = datasets.with_format(type="torch")
    print("Got dataset")

    return datasets


def get_test_data():
    # This function gets the test data using the Bad_dataset.
    # While the bad dataset is not the best dataset for testing due to its design.
    # I lack anything better, yet it will be good enough as I am only looking for the direction of the model.
    # Not the magnitude of its success.
    dataset = ImageFolder(root=TEST_DATA, transform=TRANSFORM)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


def calculate_accuracy(y_logits, y):
    y_pred = torch.sigmoid(y_logits)
    y_pred = (y_pred > .5).int()
    # print(y_pred == y)
    accuracy = (y_pred == y).sum().item()
    accuracy = accuracy/y.size(0)
    return accuracy * 100


def train(data, optimizer, loss_fn, model):
    start = time()
    model.train()

    train_loss = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(zip(data["X"], data["y"])):
        X = X.type(torch.float32)
        y = y.type(torch.float32)
        y_logits = model.forward(X)

        # There is 100% gonna be a shape error lol
        loss = loss_fn(y_logits.squeeze(dim=1), y.squeeze(dim=1))
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y.squeeze(dim=1))
        train_accuracy += accuracy

    train_accuracy = train_accuracy/len(data["X"])
    train_loss = train_loss/len(data["X"])

    end = time()
    total_time = end-start

    return train_loss, train_accuracy, total_time


def test(data, loss_fn, model):
    start = time()
    model.eval()

    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data):
            X = X.type(torch.float32)
            y = y.type(torch.float32)
            y_logits = model.forward(X)

            loss = loss_fn(y_logits.squeeze(dim=1), y)
            test_loss += loss

            accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
            test_accuracy += accuracy

        test_loss = test_loss/len(data)
        test_accuracy = test_accuracy/len(data)

        end = time()
        total_time = end-start

        return test_loss, test_accuracy, total_time


def create_and_display_dataframe(dictionary):
    # This function is used to create a dataframe of the model results using PANDASSSSSSSSSSSSSSSSS!!!!
    dataframe = DataFrame(dictionary)
    print(dataframe)
    return dataframe


def save_model(model):
    save = input("Do you want to save this model?: ")
    if not (save == "yes" or save == "Yes"):
        print("This model shall not be saved.")
        return 0
    model_path = Path(input("What file do you want to save your model in? \n"
                            "If you input a file that dose not exist. One will be created for you: "))
    model_path.mkdir(parents=True, exist_ok=True)
    model_name = input("What do you want your model dict's name to be?: ")+".pth"
    model_save_path = model_path/model_name
    print("Now downloading the model.....")
    # This will save the model dict. If you want to save the entire model then change this code to do so
    torch.save(obj=model.state_dict(), f=model_save_path)
    print("Model successfully saved! YIPPEE!!!")


def main():
    data = get_huggingface_data()
    train_data_handler = DataPreparer(data=data)
    train_data = train_data_handler.prepare_data()
    test_data = get_test_data()
    model = AdaptiveCNNModel(input_features=COLOR_CHANNELS, hidden_features=64)

    loss_fn = nn.BCEWithLogitsLoss()  # Best loss function fr
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    epochs = 5

    # Saving all the results in this dictionary for display later.
    results = {"Epoch": [], "Train loss": [], "Train accuracy": [], "Train time": [], "Test loss": [],
               "Test Accuracy": [], "Test time": []}

    print("\nStarting training process...")

    for epoch in range(epochs):
        train_loss, train_accuracy, train_time = train(train_data, optimizer, loss_fn, model)
        test_loss, test_accuracy, test_time = test(test_data, loss_fn, model)
        print(f"\nEpoch: {epoch+1}\n"
              f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.3f}% | Train time: {train_time:.3f}s\n"
              f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.3f}% | Test time: {test_time:.3f}s")
        results["Epoch"].append(epoch+1)
        results["Train loss"].append(train_loss.detach())
        results["Train accuracy"].append(round(train_accuracy, 3))
        results["Train time"].append(round(train_time, 3))
        results["Test loss"].append(test_loss.detach())
        results["Test Accuracy"].append(round(test_accuracy, 3))
        results["Test time"].append(round(test_time, 3))

    create_and_display_dataframe(results)
    save_model(model)


if __name__ == "__main__":
    main()
