import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from transformers import AutoTokenizer
import time
from pathlib import Path
from pandas import DataFrame


# 5 will be the default epoch value across all tests in test 4
EPOCH = 5
BATCH_SIZE = 32
INPUT_FEATURES = 128
OUTPUT_FEATURES = 1
HIDDEN_LAYERS = 1
HIDDEN_FEATURES = 64


def user_login():
    logged_in = input("Are you not logged in to huggingface?: ")
    while (logged_in == "yes") or (logged_in == "Yes"):
        login_token = input("Input your huggingface token in order to log in: ")
        try:
            print("Logging in...")
            login(token=login_token)
            print("You are now logged in")
            break
        except:
            print("Your token did not work. Please input a real token.")


def get_data():
    train_dataset = load_dataset("ziq/depression_tweet", split="train")
    test_dataset = load_dataset("ziq/depression_tweet", split="test")
    # Using bert-base-uncased for tokenizing the dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # This is based off of code from huggingface because I don't fully know how to tokenize stuff... yay
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, batch_size=BATCH_SIZE)
    tokenized_test_dataset = test_dataset.map(tokenize, batched=True, batch_size=BATCH_SIZE)
    tokenized_train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    tokenized_test_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    train_dataloader = DataLoader(dataset=tokenized_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=tokenized_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, test_dataloader


class BaselineModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers, hidden_features):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_features, out_features=hidden_features)
        self.output_layer = nn.Linear(in_features=hidden_features, out_features=output_features)
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features)
        )
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = self.input_layer(x)
        for _ in range(self.hidden_layers):
            x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def calculate_accuracy(y_logits, y):
    y_pred = torch.sigmoid(y_logits)
    y_pred = (y_pred > .5).int()
    # print(y_pred == y)
    accuracy = (y_pred == y).sum().item()
    accuracy = accuracy/y.size(0)
    # print(accuracy)
    return accuracy * 100


def train(dataloader, model, loss_fn, optimizer):
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    model.train()
    for batch, data in enumerate(dataloader):
        X = data["input_ids"].type(torch.float32)
        y = data["label"].type(torch.float32)
        y_logits = model.forward(X).squeeze(dim=1)
        loss = loss_fn(y_logits, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(y_logits, y)
        train_accuracy += accuracy
    train_loss = train_loss/len(dataloader)
    train_accuracy = train_accuracy/len(dataloader)
    end = time.time()
    train_time = end-start
    return train_loss, train_accuracy, train_time


def test(dataloader, model, loss_fn):
    start = time.time()
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.inference_mode():
        for batch, data in enumerate(dataloader):
            X = data["input_ids"].type(torch.float32)
            y = data["label"].type(torch.float32)
            y_logits = model.forward(X).squeeze(dim=1)
            loss = loss_fn(y_logits, y)
            test_loss += loss
            accuracy = calculate_accuracy(y_logits, y)
            test_accuracy += accuracy
        test_loss = test_loss/len(dataloader)
        test_accuracy = test_accuracy/len(dataloader)
        end = time.time()
        test_time = end-start
        return test_loss, test_accuracy, test_time


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


def create_and_display_dataframe(dictionary):
    # This function is used to create a dataframe of the model results using PANDASSSSSSSSSSSSSSSSS!!!!
    dataframe = DataFrame(dictionary)
    print(dataframe)
    return dataframe


def main():
    user_login()
    train_dataloader, test_dataloader = get_data()
    model = BaselineModel(input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES,
                          hidden_layers=HIDDEN_LAYERS, hidden_features=HIDDEN_FEATURES)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
    epochs = EPOCH
    results = {"Epoch": [], "Train loss": [], "Train accuracy": [], "Train time": [], "Test loss": [],
               "Test Accuracy": [], "Test time": []}

    for epoch in range(epochs):
        train_loss, train_accuracy, train_time = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy, test_time = test(test_dataloader, model, loss_fn)
        print(f"Epoch: {epoch+1}\nTrain loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.3f}% | "
              f"Train time: {train_time:.2f}\nTest loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.3f}% | "
              f"Test time: {test_time:.2f}")
        results["Epoch"].append(epoch+1)
        results["Train loss"].append(round(train_loss.detach().item(), 4))
        results["Train accuracy"].append(round(train_accuracy, 3))
        results["Train time"].append(round(train_time, 2))
        results["Test loss"].append(round(test_loss.detach().item(), 4))
        results["Test Accuracy"].append(round(test_accuracy, 3))
        results["Test time"].append(round(test_time, 2))
    create_and_display_dataframe(results)
    save_model(model)


if __name__ == "__main__":
    main()
