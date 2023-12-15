import time
import math
import numpy as np
from colorama import Back
import random
import matplotlib.pyplot as plt

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
    
    def train_with_loss(self, inputs, outputs, iterations, learning_rate=0.01):
        losses = []  # List to store losses over time
        for _ in range(iterations):
            # Forward pass
            predictions = self.sigmoid(np.dot(inputs, self.weights) + self.bias)

            # Loss calculation
            loss = -np.mean(outputs * np.log(predictions + 1e-15) + (1 - outputs) * np.log(1 - predictions + 1e-15))

            # Gradient descent
            errors = outputs - predictions
            dW = np.dot(inputs.T, errors).mean(axis=1)  # Calculate average gradient w.r.t. weights
            dB = errors.mean()  # Calculate average gradient w.r.t. bias

            # Update weights and biases
            self.weights += learning_rate * dW  # dW is already 1D
            self.bias += learning_rate * dB

            losses.append(loss)  # Store the loss

        return losses  # Return losses over time

    
    
class WireModel2():
    def __init__(self, dropout_rate=0.05, regularization_strength=0.01):
        self.weights = np.random.randn(4, 4)
        self.bias = np.random.randn(4)
        self.regularization_strength = regularization_strength
        self.dropout_rate = dropout_rate
        self.dropout_mask = None  # Will be used to store the dropout mask during training

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def apply_dropout(self, inputs):
        # Generate a binary mask where elements are 0 with probability dropout_rate
        self.dropout_mask = (np.random.rand(*inputs.shape) < 1 - self.dropout_rate) / (1 - self.dropout_rate)
        return inputs * self.dropout_mask

    def forward(self, inputs, training=True):
        if training:
            inputs = self.apply_dropout(inputs)
        return self.softmax(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, outputs, iterations, learning_rate=0.1):
        for _ in range(iterations):
            # Forward pass
            predictions = self.forward(inputs, training=True)

            # Gradient descent with L2 regularization
            errors = outputs - predictions
            dW = np.dot(inputs.T, errors) - self.regularization_strength * self.weights
            dB = errors.sum(axis=0)

            # Update weights and biases
            self.weights += learning_rate * dW
            self.bias += learning_rate * dB
            
    def train_with_loss(self, inputs, outputs, iterations, learning_rate=0.1):
        losses = []  # List to store losses over time
        for _ in range(iterations):
            # Forward pass
            predictions = self.forward(inputs, training=True)

            # Loss calculation
            loss = -np.sum(outputs * np.log(predictions + 1e-15)) / len(outputs)  # Add a small value to avoid log(0)

            # Gradient descent with L2 regularization
            errors = outputs - predictions
            dW = np.dot(inputs.T, errors) - self.regularization_strength * self.weights
            dB = errors.sum(axis=0)

            # Update weights and biases
            self.weights += learning_rate * dW
            self.bias += learning_rate * dB

            losses.append(loss)  # Store the loss

        return losses  # Return losses over time

    def predict(self, inputs):
        # Apply dropout during prediction to ensure consistent behavior
        inputs = self.apply_dropout(inputs)
        probabilities = self.forward(inputs, training=False)
        return np.argmax(probabilities, axis=1) + 1

    def calculate_accuracy(self, predictions, labels):
        correct = sum([1 if p == l else 0 for p, l in zip(predictions, labels)])
        return correct / len(labels) * 100


if __name__ == "__main__":
    
    num_experiments = int(input("Enter the number of experiments: "))
    print_statements = input("Do you want to print statements? (yes/no): ").lower() == "yes"
    
    total_accuracy_model1 = 0
    total_accuracy_model2 = 0
    
    losses_model1 = []
    losses_model2 = []
    

    for exp_num in range(num_experiments):
                
        dataset_size = 50 # Number of grids for training
        test_size = 50  # Number of grids for testing
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
                outputs_model2.append(wire_order[2])
                 

        inputs_model1 = np.array(inputs_model1)  
        outputs_model1 = np.array(outputs_model1) 

        inputs_model2 = np.array(inputs_model2)  
        outputs_model2 = np.array(outputs_model2) 
        # print(inputs_model2)
        # print(outputs_model2)

        # Train Model 1
        nn1 = WireModel()
        nn1.train(inputs_model1, outputs_model1, dataset_size, learning_rate=0.01)

        # Train Model 2
        nn2 = WireModel2(regularization_strength=0.01)
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
        
        # for i in range(test_size):
        #     actual_label = 'Dangerous' if test_outputs_model1[i] == 1 else 'Not Dangerous'
        #     predicted_label = 'Dangerous' if test_predictions_model1[i] == 1 else 'Not Dangerous'
        #     print(f"Grid {i + 1} (Model 1): Prediction - {predicted_label}, Actual - {actual_label}")

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
                test_outputs_model2.append(test_order[2])
                
        testinputsarray = np.array(test_inputs_model2)
            
        # print(f'THESE ARE TEST INPUTS: {testinputsarray}')
        # print(f'THESE ARE TEST OUTPUTS: {test_outputs_model2}')
            
        test_inputs_model2 = np.array(test_inputs_model2)
        
        test_predictions_model2 = nn2.predict(test_inputs_model2)

        accuracy_model2 = nn2.calculate_accuracy(test_predictions_model2, test_outputs_model2)
        total_accuracy_model2 += accuracy_model2
        
        if print_statements:
            print(f"Experiment {exp_num + 1}:")

            # Print statements for Model 1
            print("Model 1:")
            for i in range(test_size):
                actual_label = 'Dangerous' if test_outputs_model1[i] == 1 else 'Not Dangerous'
                predicted_label = 'Dangerous' if test_predictions_model1[i] == 1 else 'Not Dangerous'
                print(f"  Grid {i + 1}: Prediction - {predicted_label}, Actual - {actual_label}")

            # Print statements for Model 2
            print("Model 2:")
            for i in range(len(test_outputs_model2)):
                actual_label = f"Cut Wire {test_outputs_model2[i]}"
                predicted_label = f"Predicted to Cut Wire {test_predictions_model2[i]}"
                print(f"  Grid {i + 1}: Prediction - {predicted_label}, Actual - {actual_label}")

        
        # for i in range(len(test_outputs_model2)):
        #     actual_label = f"Cut Wire {test_outputs_model2[i]}"
        #     predicted_label = f"Predicted to Cut Wire {test_predictions_model2[i]}"
        #     print(f"Grid {i + 1} (Model 2): Prediction - {predicted_label}, Actual - {actual_label}")

    average_accuracy_model1 = total_accuracy_model1 / num_experiments
    average_accuracy_model2 = total_accuracy_model2 / num_experiments

    print(f"Average Test Accuracy (Model 1) over {num_experiments} experiments: {average_accuracy_model1}%")
    print(f"Average Test Accuracy (Model 2) over {num_experiments} experiments: {average_accuracy_model2}%")
    
    nn1 = WireModel()
    loss_model1 = nn1.train_with_loss(inputs_model1, outputs_model1, dataset_size, learning_rate=0.01)
    losses_model1.append(loss_model1)

    # Train Model 2
    nn2 = WireModel2(regularization_strength=0.01)
    outputs_model2_one_hot = np.eye(4)[outputs_model2 - 1]  # one hot encoding
    loss_model2 = nn2.train_with_loss(inputs_model2, outputs_model2_one_hot, dataset_size, learning_rate=0.01)
    losses_model2.append(loss_model2)
    
    for i, loss_values in enumerate(losses_model1):
        plt.plot(loss_values, label=f'Model 1 - Experiment {i + 1}')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over time for Model 1')
    plt.legend()
    plt.show()

    # Plot the loss over time for Model 2
    for i, loss_values in enumerate(losses_model2):
        plt.plot(loss_values, label=f'Model 2 - Experiment {i + 1}')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over time for Model 2')
    plt.legend()
    plt.show()
    
    
    
    