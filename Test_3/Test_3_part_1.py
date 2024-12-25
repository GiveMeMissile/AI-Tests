import torch
from torch import nn
import random
import time
from pathlib import Path


WINDOW_SIZE_X, WINDOW_SIZE_Y = 800, 500
NUM_TRAIN_TENSORS = 500
NUM_TEST_TENSORS = 100
LENIENCY = 20
INPUT_FEATURES = 2
OUTPUT_FEATURES = 2
USER_INPUT = True


def create_data():
    train_data = {"X": [], "y": []}
    test_data = {"X": [], "y": []}
    print("Creating training values...")
    for _ in range(NUM_TRAIN_TENSORS):
        ai_x, ai_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        goal_x, goal_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        X = torch.tensor([[ai_x, ai_y], [goal_x, goal_y]], dtype=torch.float32)
        left = 0
        right = 0
        up = 0
        down = 0
        if goal_x - LENIENCY > ai_x:
            right = 1
        elif goal_x + LENIENCY < ai_x:
            left = 1
        else:
            pass
        if goal_y - LENIENCY > ai_y:
            up = 1
        elif goal_y + LENIENCY < ai_y:
            down = 1
        else:
            pass
        y = torch.tensor([up, down, left, right], dtype=torch.float32)
        train_data["X"].append(X)
        train_data["y"].append(y)
    print("Training values created.")

    print("Creating testing values...")
    for _ in range(NUM_TEST_TENSORS):
        ai_x, ai_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        goal_x, goal_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        X = torch.tensor([[ai_x, ai_y], [goal_x, goal_y]], dtype=torch.float32)
        left = 0
        right = 0
        up = 0
        down = 0
        if goal_x > ai_x:
            right = 1
        elif goal_x < ai_x:
            left = 1
        else:
            pass
        if goal_y > ai_y:
            down = 1
        elif goal_y < ai_y:
            up = 1
        else:
            pass
        y = torch.tensor([up, down, left, right], dtype=torch.float32)
        test_data["X"].append(X)
        test_data["y"].append(y)
    print("Testing values created.")
    return train_data, test_data


class AIModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers, hidden_features):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_features, out_features=hidden_features)
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features)
        )
        self.output_layer = nn.Linear(in_features=hidden_features, out_features=output_features)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.hidden_layers):
            x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def get_user_input():
    epochs = 5
    hidden_layers = 1
    hidden_features = 64
    learning_rate = 0.001
    if not USER_INPUT:
        print("User input is not activated.")
        print(f"User input has not been collected. Using default values \n"
              f"hidden layers: {hidden_layers}, hidden features: {hidden_features}, learning rate: {learning_rate}, epochs: {epochs}")
        return hidden_layers, hidden_features, learning_rate, epochs
    print("User input is active, please answer the following questions in order to customize the AI. "
          "If you don't want user input next time then turn 'USER_INPUT' to False.")
    while USER_INPUT:
        try:
            hidden_layers = int(input("How many hidden layers do you want?: "))
            break
        except ValueError:
            print("You didn't input a integer, please try again.")
    while USER_INPUT:
        try:
            hidden_features = int(input("How many hidden features/neurons do you want?: "))
            break
        except ValueError:
            print("You didn't input a integer, please try again.")
    while USER_INPUT:
        try:
            learning_rate = float(input("What learning rate do you want (float)?: "))
            break
        except ValueError:
            print("You didn't input a float, please try again.")
    while USER_INPUT:
        try:
            epochs = int(input("How many epochs do you want?: "))
            break
        except ValueError:
            print("You didn't input a integer, please try again.")
    print(f"User input has been collected. \n hidden layers: {hidden_layers}, "
          f"hidden features: {hidden_features}, learning rate: {learning_rate}, epochs: {epochs}")
    return hidden_layers, hidden_features, learning_rate, epochs


def calculate_acc(y_logits, y):
    # This function calculates the accuracy of the AI model
    y_pred = y_logits.sigmoid()
    # This line of code below converts the tensors into booleans, compares the two,
    # and then turns the tensor of booleans into a list that can be iterated through
    acc_list = (y_pred.round() == y).tolist()
    acc = 0
    # This for loop below iterates through the list in order to calculate the accuracy
    for equal in acc_list:
        if equal:
            acc += 25
    return acc


def train(loss_fn, optimizer, model, train_dataset):
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    for _, (X, y) in enumerate(zip(train_dataset["X"], train_dataset["y"])):
        y_logits = model.forward(X)
        y_logits = torch.reshape(y_logits, (-1,))
        loss = loss_fn(y_logits, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = calculate_acc(y_logits, y)
        train_accuracy += acc
    print(train_accuracy)
    train_loss = train_loss/NUM_TRAIN_TENSORS
    train_accuracy = train_accuracy/NUM_TRAIN_TENSORS
    end = time.time()
    total_train_time = end-start
    return train_loss, train_accuracy, total_train_time


def test(loss_fn, model, test_dataset):
    test_loss = 0
    test_accuracy = 0
    start = time.time()
    with torch.inference_mode():
        for _, (X, y) in enumerate(zip(test_dataset["X"], test_dataset["y"])):
            y_logits = model.forward(X)
            y_logits = torch.reshape(y_logits, (-1,))
            loss = loss_fn(y_logits, y)
            test_loss += loss
            acc = calculate_acc(y_logits, y)
            test_accuracy += acc
        test_loss = test_loss/NUM_TEST_TENSORS
        test_accuracy = test_accuracy/NUM_TEST_TENSORS
        end = time.time()
        total_test_time = end-start
        return test_loss, test_accuracy, total_test_time


def save_model(model):
    if not USER_INPUT:
        # You can change this code below in order to save the model if you want.
        print("No save -_-")
        return 0
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


# I am using a main function for once.
def main():
    train_dataset, test_dataset = create_data()
    hidden_layers, hidden_features, learning_rate, epochs = get_user_input()
    model = AIModel(input_features=INPUT_FEATURES, hidden_features=hidden_features,
                    hidden_layers=hidden_layers, output_features=OUTPUT_FEATURES)
    loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss is used for multi label classification
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss, train_accuracy, total_train_time = train(loss_fn, optimizer, model, train_dataset)
        test_loss, test_accuracy, total_test_time = test(loss_fn, model, test_dataset)
        print(f"\nEpoch: {epoch}\n"
              f"train loss: {train_loss:.4f} | train accuracy: {train_accuracy:.3f}% | train time: {total_train_time:.2f}."
              f"\ntest loss: {test_loss:.4f} | test accuracy: {test_accuracy:.3f}% | test time: {total_test_time:.2f}.")
    save_model(model)


if __name__ == "__main__":
    main()
