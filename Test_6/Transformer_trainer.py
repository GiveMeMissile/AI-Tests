# This is where I will be training the transformer model to detect depression from text.

# Importing the important libaries. ALL OF THEM WILL BE USED!!! (wow)
import torch
from torch import nn
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
import time
from pathlib import Path
from pandas import DataFrame
import math
from random import shuffle


# Using the best base uncase for tokenization. 
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
EPOCH = 25
BATCH_SIZE = 64
MAX_LENGTH = 256  # The maxiumum number of tokens that a text can have. Not too much actually.
INPUT_FEATURES = MAX_LENGTH
OUTPUT_FEATURES = 1
HIDDEN_LAYERS = 6  # The default number of transformer layers.
HIDDEN_FEATURES = 512  # The default number of hidden features
NUM_TOKENS = len(TOKENIZER)  # All of the possible tokens. Used in embedding layer.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device agnostic code.

# Below are several different number of datasets that will be used for training.
DATASET_1 = "thePixel42/depression-detection"
DATASET_1_TEST = True
DATASET_2 = "joangaes/depression"
DATASET_2_TEST = False
DATASET_3 = "mangoesai/DepressionDetection"
DATASET_3_TEST = True
DATASET_4 = "ziq/depression_tweet"
DATASET_4_TEST = True


def set_data_device(device, X, y):
    # Sets the data's device and datatype

    X = X.to(device).type(torch.long)
    y = y.to(device).type(torch.float)
    return X, y


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


# A class that manages multiple datasets
class DatasetManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # The lists below save the entered datasets allowing for easy data manipulation.
        self.dataset_storage_train = []
        self.dataset_storage_test = []
        self.tokenized_dataset_train = []
        self.tokenized_dataset_test = []
        self.train_dict = {}
        self.test_dict = {}


    def load_huggingface_data(self, dataset, contains_test_files):
        # This function loads the dataset from huggingface.

        if contains_test_files:
            train_dataset = load_dataset(dataset, split="train")
            test_dataset = load_dataset(dataset, split="test")
        else:
            # splits the train file into both train and test data.

            train_test_split= load_dataset(dataset, split="train").train_test_split(test_size=.2)
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]

        self.dataset_storage_train.append(train_dataset)
        self.dataset_storage_test.append(test_dataset)
        
        return train_dataset, test_dataset

    def tokenize(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    def bugfix_tokenize(self, example):
        return self.tokenizer(example["clean_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    def tokenize_dataset(self, train_dataset, test_dataset):
        # This function tokenizes the inputted datasets and converts them to torch.
        try:
            tokenized_train_dataset = train_dataset.map(self.tokenize, batched=True, batch_size=BATCH_SIZE)
            tokenized_test_dataset = test_dataset.map(self.tokenize, batched=True, batch_size=BATCH_SIZE)
        except Exception:
            tokenized_train_dataset = train_dataset.map(self.bugfix_tokenize, batched=True, batch_size=BATCH_SIZE)
            tokenized_test_dataset = test_dataset.map(self.bugfix_tokenize, batched=True, batch_size=BATCH_SIZE)

        try:
            tokenized_train_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"]
            )
            tokenized_test_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"]
            )
        
        except Exception:
            tokenized_train_dataset.set_format(
                type="torch",
                columns=["clean_text", "is_depression", "input_ids", "token_type_ids", "attention_mask"]
            )
            tokenized_test_dataset.set_format(
                type="torch",
                columns=["clean_text", "is_depression", "input_ids", "token_type_ids", "attention_mask"]
            )

        self.tokenized_dataset_train.append(tokenized_train_dataset)
        self.tokenized_dataset_test.append(tokenized_test_dataset)

        return tokenized_train_dataset, tokenized_test_dataset
    
    def tokenize_all_saved_datasets(self):
        # This function tokenizes all the saved datasets. This makes it easier to tokenize all the data.

        for (train, test) in zip(self.dataset_storage_train, self.dataset_storage_test):
            self.tokenize_dataset(train, test)

    def merge_tokenized_datasets(self):
        # This function merges all of the datasets into a dict containing the X(input) and y(target).
        # Test and Training will not be merged because we need to keep test seperate from train so we can evaluate the Transformer model.

        # Initial values that will be used to store the inputs(X) and outputs(y).
        test_X = []
        test_y = []
        train_X = []
        train_y = []

        for train_dataset in self.tokenized_dataset_train:
            train_X += train_dataset["input_ids"]  # input
            try:
                train_y += train_dataset["label"]  # output
            except Exception:
                train_y += train_dataset["is_depression"]  # also output

        for test_dataset in self.tokenized_dataset_test:
            test_X += test_dataset["input_ids"]  # input
            try:
                test_y += test_dataset["label"]  # output
            except Exception:
                test_y += test_dataset["is_depression"]  # also output


        train_dataset = {"X": train_X, "y": train_y}
        test_dataset = {"X": test_X, "y": test_y}

        self.train_dict = train_dataset
        self.test_dict = test_dataset

        return train_dataset, test_dataset
    
    def shuffle_data(self, dataset):
        # As the title suggests this function shuffles the data

        merged_data = list(zip(dataset["X"], dataset["y"]))
        shuffle(merged_data)
        dataset["X"], dataset["y"] = zip(*merged_data)
        return dataset
    
    def apply_shuffle(self):
        # This function applies the shuffle_data function to the test and train dict.

        self.train_dict = self.shuffle_data(self.train_dict)
        self.test_dict = self.shuffle_data(self.test_dict)

    def batch_data(self, dataset=None):
        # This function batches the data in order to train a neural network.

        batch = 1
        X_batch = []
        y_batch = []
        X_list = []
        y_list = []
        for i, (text, category) in enumerate(zip(dataset["X"], dataset["y"])):
            X_batch.append(text)
            y_batch.append(category)
            if i+1 == BATCH_SIZE*batch or len(dataset["X"]) == i+1:
                batch += 1
                X = torch.stack(X_batch, dim=0)
                y = torch.stack(y_batch, dim=0)
                X_batch.clear()
                y_batch.clear()
                X_list.append(X)
                y_list.append(y)
        dataset = {"X": X_list, "y": y_list}
        return dataset
    
    def apply_batch(self):
        self.train_dict = self.batch_data(self.train_dict)
        self.test_dict = self.batch_data(self.test_dict)

    def prepare_all_loaded_data(self):
        print("Preparing data...")
        self.tokenize_all_saved_datasets()
        self.merge_tokenized_datasets()
        self.apply_shuffle()
        self.apply_batch()
        print("Data sucessfully prepared")
        return self.train_dict, self.test_dict


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

    def forward(self, x, padding_attention_mask=None):
        x = x.to(DEVICE)
        x = self.embedding(x.long()) * math.sqrt(self.hidden_features)
        x = self.positional_encoder(x)

        x = x.permute(1, 0, 2)

        try:
            x = self.transformer(x, src_key_padding_mask=padding_attention_mask)
        except Exception as e:
            print(f"attention mask: {padding_attention_mask}")
            print(f"x: {x}")
            print("shape of x: ", x.shape)
            # print(f"shape of padding_attention_mask: {padding_attention_mask.shape}")
            raise e
        x = self.transformer(x, src_key_padding_mask=padding_attention_mask)

        x = self.output_layer(x.mean(0))
        return x


def calculate_accuracy(y_logits, y):
    y_pred = torch.sigmoid(y_logits)
    y_pred = (y_pred > .5).int()
    accuracy = (y_pred == y).sum().item()
    accuracy = accuracy/y.size(0)
    return accuracy * 100


def create_attention_mask(X):
    # This function creates an attention mask for the transformer model.
    # This mask is used to mask padding tokens (0s) in the input.

    attention_mask = (X != 0)
    attention_mask = attention_mask.to(DEVICE)
    attention_mask = attention_mask.permute(1, 0)
    return attention_mask


def train(dataloader, model, loss_fn, optimizer):
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    model.train()
    for batch, (X, y) in enumerate(zip(dataloader["X"], dataloader["y"])):
        X, y = set_data_device(DEVICE, X, y)

        attention_mask = create_attention_mask(X)

        y_logits = model.forward(X, padding_attention_mask=attention_mask)
        loss = loss_fn(y_logits.squeeze(dim=1), y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
        train_accuracy += accuracy
    train_loss = train_loss/len(dataloader["X"])
    train_accuracy = train_accuracy/len(dataloader["X"])
    end = time.time()
    train_time = end-start
    return train_loss, train_accuracy, train_time


def test(dataloader, model, loss_fn):
    start = time.time()
    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch, (X, y) in enumerate(zip(dataloader["X"], dataloader["y"])):
            X, y = set_data_device(DEVICE, X, y)

            attention_mask = None # create_attention_mask(X)

            y_logits = model.forward(X, padding_attention_mask=attention_mask)
            loss = loss_fn(y_logits.squeeze(dim=1), y)
            # print(f"loss: {loss}")
            # print(f"y_logits: {y_logits}")
            test_loss += loss
            accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
            test_accuracy += accuracy
        test_loss = test_loss/len(dataloader["X"])
        test_accuracy = test_accuracy/len(dataloader["X"])
        end = time.time()
        test_time = end-start
        return test_loss, test_accuracy, test_time


def save_model(model):
    # function the automatically saves the model

    model = model.to(DEVICE)

    print("Saving model...")
    model_path = Path("Depression_detecting_models")
    model_path.mkdir(parents=True, exist_ok=True)
    model_name = "Model_001.pth"
    model_save_path = model_path/model_name
    torch.save(model, model_save_path)
    model_state_dict_name = "Model_001_state_dict.pth"
    dict_path = model_path/model_state_dict_name
    torch.save(model.state_dict, dict_path)
    print("Model has been saved!")


def create_and_display_dataframe(dictionary):
    # This function is used to create a dataframe of the model results using PANDASSSSSSSSSSSSSSSSS!!!!
    dataframe = DataFrame(dictionary)
    print(dataframe)
    return dataframe


def main():
    print("Loading data...")
    data_manager = DatasetManager(TOKENIZER)
    data_manager.load_huggingface_data(DATASET_1, DATASET_1_TEST)
    data_manager.load_huggingface_data(DATASET_2, DATASET_2_TEST)
    data_manager.load_huggingface_data(DATASET_3, DATASET_3_TEST)
    data_manager.load_huggingface_data(DATASET_4, DATASET_4_TEST)

    train_dataloader, test_dataloader = data_manager.prepare_all_loaded_data()

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
