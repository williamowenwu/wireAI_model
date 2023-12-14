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
        clipped_x = np.clip(x, -500, 500)  # Clip x to avoid overflow and underflow issues
        return 1 / (1 + np.exp(-clipped_x))

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
    
    
class WireModel2():
    def __init__(self):
        self.weights = np.random.randn(4, 4)  # 4 inputs for wire order, 4 neurons in output layer to decide which one to cut
        self.bias = np.random.randn(4)
    
    
    #used for multiclassification
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


    def forward(self, inputs):
        return self.softmax(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, outputs, iterations, learning_rate=0.01):
        for _ in range(iterations):
            # Forward pass
            predictions = self.forward(inputs)

            # Gradient descent
            errors = outputs - predictions
            dW = np.dot(inputs.T, errors)  # Gradient w.r.t. weights
            dB = errors.sum(axis=0)  # Gradient w.r.t. bias

            # Update weights and biases
            self.weights += learning_rate * dW
            self.bias += learning_rate * dB

    def predict(self, inputs):
        probabilities = self.forward(inputs)
        return np.argmax(probabilities, axis=1) + 1  # Add 1 to convert from 0-indexed to 1-indexed

    def calculate_accuracy(self, predictions, labels):
        correct = sum([1 if p == l else 0 for p, l in zip(predictions, labels)])
        return correct / len(labels) * 100

if __name__ == "__main__":
    num_experiments = 1
    total_accuracy_model1 = 0
    total_accuracy_model2 = 0
    

    for _ in range(num_experiments):
                
        dataset_size = 1000  # Number of grids for training
        test_size = 500  # Number of grids for testing
        inputs_model1 = []
        outputs_model1 = []

        inputs_model2 = []
        outputs_model2 = []

        for _ in range(dataset_size):
            grid = WireGrid()
            grid.generate_Grid()
            wire_order = grid.get_wire_order()
            danger = grid.is_dangerous()

            inputs_model1.append(wire_order)
            outputs_model1.append([danger])

            if danger:
                inputs_model2.append(wire_order)
                outputs_model2.append(grid.used_color.index('Y') + 1)  

        inputs_model1 = np.array(inputs_model1)  
        outputs_model1 = np.array(outputs_model1) 

        inputs_model2 = np.array(inputs_model2)  
        outputs_model2 = np.array(outputs_model2) 

        # Train Model 1
        nn1 = WireModel()
        nn1.train(inputs_model1, outputs_model1, dataset_size, learning_rate=0.01)

        # Train Model 2
        nn2 = WireModel2()
        outputs_model2_one_hot = np.eye(4)[outputs_model2 - 1] #one hot encoding which represents output labels for each color
        nn2.train(inputs_model2, outputs_model2_one_hot, dataset_size, learning_rate=0.01)

        test_inputs_model1 = []
        test_outputs_model1 = []

        for _ in range(test_size):
            test_grid = WireGrid()
            test_grid.generate_Grid()
            test_order = test_grid.get_wire_order()
            test_danger = test_grid.is_dangerous()

            test_inputs_model1.append(test_order)
            test_outputs_model1.append(test_danger)

        test_inputs_model1 = np.array(test_inputs_model1)
        test_predictions_model1 = nn1.predict(test_inputs_model1)

        accuracy_model1 = nn1.calculate_accuracy(test_predictions_model1, test_outputs_model1)
        total_accuracy_model1 += accuracy_model1

        # Generate test data for Model 2
        test_inputs_model2 = []
        test_outputs_model2 = []

        for _ in range(test_size):
            test_grid = WireGrid()
            test_grid.generate_Grid()
            test_order = test_grid.get_wire_order()
            test_danger = test_grid.is_dangerous()

            if test_danger:
                test_inputs_model2.append(test_order)
                test_outputs_model2.append(test_grid.used_color.index('Y') + 1)  # Index of 'Y' color + 1

        test_inputs_model2 = np.array(test_inputs_model2)
        test_predictions_model2 = nn2.predict(test_inputs_model2)

        accuracy_model2 = nn2.calculate_accuracy(test_predictions_model2, test_outputs_model2)
        total_accuracy_model2 += accuracy_model2

    average_accuracy_model1 = total_accuracy_model1 / num_experiments
    average_accuracy_model2 = total_accuracy_model2 / num_experiments

    print(f"Average Test Accuracy (Model 1) over {num_experiments} experiments: {average_accuracy_model1}%")
    print(f"Average Test Accuracy (Model 2) over {num_experiments} experiments: {average_accuracy_model2}%")
