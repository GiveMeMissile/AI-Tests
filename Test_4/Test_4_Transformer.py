import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from transformers import AutoTokenizer
import time
from pathlib import Path
from pandas import DataFrame
import math


# 5 will be the default epoch value across all tests
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
EPOCH = 5
BATCH_SIZE = 32
MAX_LENGTH = 64
INPUT_FEATURES = MAX_LENGTH
OUTPUT_FEATURES = 1
HIDDEN_LAYERS = 2
HIDDEN_FEATURES = 64
DATASET = "thePixel42/depression-detection"
NUM_TOKENS = len(TOKENIZER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


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
    print("Processing data...")
    train_dataset = load_dataset(DATASET, split="train")
    test_dataset = load_dataset(DATASET, split="test")
    # Using bert-base-uncased for tokenizing the dataset

    # This is based off of code from huggingface because I don't fully know how to tokenize stuff... yay
    def tokenize(example):
        return TOKENIZER(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

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
    print("Data processed")
    return train_dataloader, test_dataloader


class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_length, dimensions)
        positions_list = torch.arange(0, max_length, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0))/dimensions)

        pos_encoding[:, 0::2] = torch.sin(positions_list*division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[:x.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, num_tokens, hidden_features, num_heads, num_layers, dropout, output):
        super(TransformerModel, self).__init__()

        self.hidden_features = hidden_features

        self.positional_encoder = PositionalEncoding(hidden_features, dropout, max_length=MAX_LENGTH)

        self.embedding = nn.Embedding(num_tokens, hidden_features)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=num_heads,
            dim_feedforward=HIDDEN_FEATURES,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_features, output)

    def forward(self, x, attention_mask=None):
        x = x.to(DEVICE)
        x = self.embedding(x.long()) * math.sqrt(self.hidden_features)
        x = self.positional_encoder(x)

        x = x.permute(1, 0, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.permute(1, 0)
            attention_mask = attention_mask.to(dtype=torch.bool)

        x = self.transformer(x, src_key_padding_mask=attention_mask)

        x = self.output_layer(x.mean(0))
        x = x.to("cpu")
        return x


def calculate_accuracy(y_logits, y):
    y_pred = torch.sigmoid(y_logits)
    y_pred = (y_pred > .5).int()
    accuracy = (y_pred == y).sum().item()
    accuracy = accuracy/y.size(0)
    return accuracy * 100


def train(dataloader, model, loss_fn, optimizer):
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    model.train()
    for batch, data in enumerate(dataloader):
        X = data["input_ids"].type(torch.long)
        y = data["label"].type(torch.float)

        attention_mask = None

        y_logits = model.forward(X, attention_mask=attention_mask)
        loss = loss_fn(y_logits.squeeze(dim=1), y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
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
            X = data["input_ids"].type(torch.long)
            y = data["label"].type(torch.float)
            
            attention_mask = None

            y_logits = model.forward(X, attention_mask=attention_mask)
            loss = loss_fn(y_logits.squeeze(dim=1), y)
            test_loss += loss
            accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
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
    model = TransformerModel(
        num_tokens=NUM_TOKENS,
        hidden_features=HIDDEN_FEATURES,
        num_heads=8,
        num_layers=HIDDEN_LAYERS,
        dropout=.2,
        output=OUTPUT_FEATURES
    )
    model = model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    epochs = EPOCH
    results = {"Epoch": [], "Train loss": [], "Train accuracy": [], "Train time": [], "Test loss": [],
               "Test Accuracy": [], "Test time": []}
    print("Starting the training process...")

    for epoch in range(epochs):
        train_loss, train_accuracy, train_time = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy, test_time = test(test_dataloader, model, loss_fn)
        print(f"\nEpoch: {epoch+1}\nTrain loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.3f}% | "
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
