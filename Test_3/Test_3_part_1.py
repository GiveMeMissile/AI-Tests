import torch
from torch import nn
import random


WINDOW_SIZE_X, WINDOW_SIZE_Y = 800, 500
NUM_TRAIN_TENSORS = 500
NUM_TEST_TENSORS = 100
LENIENCY = 20
INPUT_FEATURES = 4
OUTPUT_FEATURES = 4
USER_INPUT = TrueTraining_Game_AI.pyTraining_Game_AI.py


def create_data():
    train_data = {"X": [], "y": []}
    test_data = {"X": [], "y": []}
    print("Creating training values...")
    for _ in range(NUM_TRAIN_TENSORS):
        ai_x, ai_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        goal_x, goal_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        X = torch.tensor([[ai_x, ai_y], [goal_x, goal_y]])
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
        y = torch.tensor([up, down, left, right])
        train_data["X"].append(X)
        train_data["y"].append(y)
    print("Training values created.")

    print("Creating testing values...")
    for _ in range(NUM_TEST_TENSORS):
        ai_x, ai_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        goal_x, goal_y = random.randint(0, WINDOW_SIZE_X), random.randint(0, WINDOW_SIZE_Y)
        X = torch.tensor([[ai_x, ai_y], [goal_x, goal_y]])
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
        y = torch.tensor([up, down, left, right])
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


def train(loss_fn, optimizer, model, train_dataset):
    pass


def test(loss_fn, model, test_dataset):
    pass


# I am using a main function for once.
def main():
    train_dataset, test_dataset = create_data()
    hidden_layers, hidden_features, learning_rate, epochs = get_user_input()
    model = AIModel(input_features=INPUT_FEATURES, hidden_features=hidden_features,
                    hidden_layers=hidden_layers, output_features=OUTPUT_FEATURES)
    loss_fn = nn.BCEWithLogitsLoss  # BCEWithLogitsLoss is used for multi label classification
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train(loss_fn, optimizer, model, train_dataset)
        test(loss_fn, model, test_dataset)


if __name__ == "__main__":
    main()
