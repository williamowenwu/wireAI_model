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
    def __init__(self) -> None:
        self.weights1 = np.random.randn(4, 10)  # 4 inputs for wire order, 10 neurons in hidden layer
        self.weights2 = np.random.randn(10, 1)
        self.bias1 = np.random.randn(10)
        self.bias2 = np.random.randn(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return output

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            # Forward pass
            hidden = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
            output = self.sigmoid(np.dot(hidden, self.weights2) + self.bias2) # value between 1 and 0

            print(f"hidden: {hidden}")
            print(f"Output: {output}")

            # Backpropagation
            output_error = outputs - output # 0 ideal for no cost correlation error
            output_delta = output_error * self.sigmoid_derivative(output)

            print(f"output error: {output_error}")
            print(f"output_delta: {output_delta}")
            print(f"outputs: {outputs}")
            
            # print(f"hidden error: {hidden_error}")
            # print(f"hidden_delta: {hidden_delta}")
            
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

            # Update weights and biases
            self.weights2 += hidden.T.dot(output_delta)
            self.bias2 += np.sum(output_delta, axis=0)

            self.weights1 += inputs.T.dot(hidden_delta)
            self.bias1 += np.sum(hidden_delta, axis=0)

    def predict(self, inputs):
        output = self.forward(inputs)
        print(output)
        return [1 if o > 0.5 else 0 for o in output]

    def calculate_accuracy(self, predictions, labels):
        correct = sum([1 if p == l else 0 for p, l in zip(predictions, labels)])
        return correct / len(labels) * 100

if __name__ == "__main__":
    dataset_size = 1  # Number of grids
    inputs = []
    outputs = []

    for _ in range(dataset_size):
        grid = WireGrid()
        grid.generate_Grid()
        grid.print_grid()
        print(f"True label: {grid.is_dangerous()}")
        time.sleep(1)
        wire_order = grid.get_wire_order()
        danger = grid.is_dangerous()

        inputs.append(wire_order)
        outputs.append([danger])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # Train the neural network
    nn = WireModel()
    nn.train(inputs, outputs, iterations=1)

    # Predict new grid
    new_grid = WireGrid()
    new_grid.generate_Grid()
    prediction = nn.predict([new_grid.get_wire_order()])

    print(wire_order)
    print(f"Prediction: Dangerous" if prediction[0] == 1 else "Not Dangerous")
    print(nn.calculate_accuracy(prediction, outputs))
    

    