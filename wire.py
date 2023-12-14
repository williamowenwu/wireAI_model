import time
import math
import numpy as np
from colorama import Back
import random

class WireGrid():
    def __init__(self) -> None:
        self.D = 20 # 20 x 20  Dimension of ship
        self.grid = []

        # colors possible in the grid the colors being used will be indices 0 - 3
        self.colors = ["R", "B", "Y", "G"]
        self.used_color = [] # also functions as wire color order

    def generate_Grid(self) -> None:
        self.grid = [["W" for _ in range(self.D)] for _ in range(self.D)]
        row_col = random.randint(0, 1) # randomly row or col
        used_row = []
        used_col = []

        for _ in range(len(self.colors)):
            if row_col == 1:
                while True:
                    row = random.randint(0, self.D - 1)
                    if row not in used_row:
                        color = self.choose_unused_color()
                        self.used_color.append(color)
                        self.grid[row] = [color] * self.D
                        row_col = 0
                        used_row.append(row)
                        break
            else:
                while True:
                    col = random.randint(0, self.D - 1)
                    if col not in used_col:
                        color = self.choose_unused_color()
                        self.used_color.append(color)
                        for x in range(self.D):
                            self.grid[x][col] = color
                        row_col = 1
                        used_col.append(col)
                        break

    def choose_unused_color(self):
        while True:
            color = random.choice(self.colors)
            if color not in self.used_color:
                return color

    def get_wire_order(self):
            color_map = {'R': 1, 'B': 2, 'Y': 3, 'G': 4} # R Y G B 
            wire_order = [color_map[color] for color in self.used_color]
            return wire_order
    
    # returns if label is dangerous or not --> 1 = dangerous, 0 = not
    # returns if label is dangerous or not --> 1 = dangerous, 0 = not
    def is_dangerous(self):
        found_yellow = False
        for i in self.used_color:
            if not found_yellow:
                if i == 'R':
                    return 1
                elif i == 'Y':
                    found_yellow = True
        return 0


    def print_grid(self) -> None:
        for x in range(grid.D):
            print(" ", end="")
            for y in range(grid.D):
                if grid.grid[x][y] == "W": 
                    print(Back.WHITE + " ", end = " ")
                elif grid.grid[x][y] == "G": 
                    print(Back.GREEN + " ", end = " ")
                elif grid.grid[x][y] == "Y": 
                    print(Back.YELLOW + " ", end = " ") 
                elif grid.grid[x][y] == "B": 
                    print(Back.CYAN + " ", end = " ") 
                elif grid.grid[x][y] == "R": 
                    print(Back.RED + " ", end = " ")    
            print(Back.RESET) 

class WireModel():
    def __init__(self):
        self.weights = np.random.randn(4)  # 4 inputs for wire order
        self.bias = np.random.randn()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, outputs, iterations, learning_rate=0.01):
        for _ in range(iterations):
            # Forward pass
            predictions = self.forward(inputs)

            # Gradient descent
            errors = outputs - predictions
            dW = np.dot(inputs.T, errors).mean(axis=1)  # Calculate average gradient w.r.t. weights
            dB = errors.mean()  # Calculate average gradient w.r.t. bias

            # Update weights and biases
            self.weights += learning_rate * dW  # dW is already 1D
            self.bias += learning_rate * dB

    def predict(self, inputs):
        probabilities = self.forward(inputs)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def calculate_accuracy(self, predictions, labels):
        correct = sum([1 if p == l else 0 for p, l in zip(predictions, labels)])
        return correct / len(labels) * 100

if __name__ == "__main__":
    dataset_size = 50  # Number of grids for training
    test_size = 20  # Number of grids for testing
    inputs = []
    outputs = []

    # Generate training data
    for _ in range(dataset_size):
        grid = WireGrid()
        grid.generate_Grid()
        wire_order = grid.get_wire_order()
        danger = grid.is_dangerous()

        inputs.append(wire_order)
        outputs.append([danger])

    inputs = np.array(inputs)  # Should be shape (dataset_size, 4)
    outputs = np.array(outputs)  # Should be shape (dataset_size, 1)

    # Train the neural network
    nn = WireModel()
    nn.train(inputs, outputs, dataset_size, learning_rate=0.01)

    # Generate test data
    test_inputs = []
    test_outputs = []

    for _ in range(test_size):
        test_grid = WireGrid()
        test_grid.generate_Grid()
        test_order = test_grid.get_wire_order()
        test_danger = test_grid.is_dangerous()

        test_inputs.append(test_order)
        test_outputs.append(test_danger)

    test_inputs = np.array(test_inputs)
    test_predictions = nn.predict(test_inputs)

    # Print predictions and correct labels for each test data
    for i in range(test_size):
        actual_label = 'Dangerous' if test_outputs[i] == 1 else 'Not Dangerous'
        predicted_label = 'Dangerous' if test_predictions[i] == 1 else 'Not Dangerous'
        print(f"Grid {i+1}: Prediction - {predicted_label}, Actual - {actual_label}")

    # Calculate and print accuracy on the test set
    accuracy = nn.calculate_accuracy(test_predictions, test_outputs)
    print(f"Test Accuracy: {accuracy}%")