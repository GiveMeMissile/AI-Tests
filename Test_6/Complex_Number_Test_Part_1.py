# What am I doing with my life ( – ⌓ – )

import torch
import torch.nn as nn
import random
import time

device = 'cuda' if torch.cuda.is_available else 'cpu'  # Device agnostic code. A classic.
EPOCHS = 100
BATCH_SIZE = 128
PATH = "Models/complex_model_001.pth"

def create_complex_tensors(number_of_values, batch):
    # The FUN PART. Creating a dataset of complex numbers! 

    func = lambda z : (4.3301 + 2.5j)*z + (5 + 10j)
    data = {"X": [], "y": []}

    for i in range(int(number_of_values/batch)):
        x_batch = []
        y_batch = []
        for _ in range(batch):
            x = random.randint(1, 100) + random.randint(1, 100)*1j  # Why is the imaginary number in python 'j'. 'i' IS RIGHT THERE!!!
            x_batch.append(torch.tensor(x, dtype=torch.complex64))
            y = func(x)
            y_batch.append(torch.tensor(y, dtype = torch.complex64))
        x = torch.stack(x_batch, dim=0).unsqueeze(1)
        y = torch.stack(y_batch, dim=0).unsqueeze(1)
        data["X"].append(x)
        data["y"].append(y)

    # Returns a dictionary with inputs "X" and outputs "y".

    return data 


class BasicNN(nn.Module):
    # Simple neural network. SIMPLE!!!

    def __init__(self):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(1, 64, dtype=torch.complex64)
        self.fc2 = nn.Linear(64, 1, dtype=torch.complex64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def calculate_accuracy(y_pred, y):
    # Calculating the accuracy with cool maths.

    real_correct = 0
    imag_correct = 0
    for i in range(y.size(0)):
        r = 100 - abs(abs(y_pred[i].real.item()-y[i].real.item())/((y[i].real.item()+y_pred[i].real.item())/2)*100)  # PEAK MATH EQUATION!!!
        im = 100 - abs(abs(y_pred[i].imag.item()-y[i].imag.item())/((y[i].imag.item()+y_pred[i].imag.item())/2)*100)  # Seriously this is a peak math equaton
        real_correct += r
        imag_correct += im
    real_correct = real_correct/y.size(0)
    imag_correct = imag_correct/y.size(0)
    return (real_correct + imag_correct)/2


def train(loss_fn, model, optimizer, data):
    start = time.time()
    model.train()

    train_loss = 0
    train_accuracy = 0

    for x, y in zip(data["X"], data["y"]):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_accuracy += calculate_accuracy(y_pred, y)
    end = time.time()
    train_loss = train_loss/len(data["X"])
    train_accuracy = train_accuracy/len(data["X"])

    return train_loss, train_accuracy, end-start

def test(loss_fn, model, data):
    start = time.time()
    model.eval()

    test_loss = 0
    test_accuracy = 0
    with torch.inference_mode():
        for x, y in zip(data["X"], data["y"]):
            x, y = x.to(device), y.to(device)
            y_pred  = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(y_pred, y)
        end = time.time()
        test_loss = test_loss/len(data["X"])
        test_accuracy = test_accuracy/len(data["X"])
        return test_loss, test_accuracy, end-start

def save_model(model, path):
    # Save the model to a file.
    print("Saving model...")
    torch.save(model.state_dict(), path)
    print("Model saved.")


def main():
    train_data = create_complex_tensors(10*(32**2), BATCH_SIZE)
    test_data = create_complex_tensors((32**2), BATCH_SIZE)
    epochs = EPOCHS

    model = BasicNN().to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train_loss, train_accuracy, train_time = train(loss_fn, model, optimizer, train_data)
        test_loss, test_accuracy, test_time = test(loss_fn, model, test_data)
        print(f"Epoch: {epoch+1} \nTrain Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.3f}% | Train Time: {train_time:.3f}" 
              f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.3f}% | Test Time: {test_time:.3f}")
        
        # Recreate the data every epoch because I can.
        train_data = create_complex_tensors(10*(32**2), BATCH_SIZE)
        test_data = create_complex_tensors((32**2), BATCH_SIZE)

    save_model(model, PATH)

if __name__ == "__main__":
    main()
