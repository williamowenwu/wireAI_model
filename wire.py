import math
from colorama import Back
import random


class Wire_Grid():
    def __init__(self) -> None:
        self.D = 20 # 20 x 20  Dimension of ship
        self.grid = []

        # colors possible in the grid the colors being used will be indices 0 - 3
        self.colors = ["R", "B", "Y", "G"]
        self.used = []
        self.wire_order = []

    def generate_Grid(self) -> None:
        self.grid = [["W" for _ in range(self.D)] for _ in range(self.D)]
        # Choose either a row or a column randomly
        row_col = random.randint(0, 1)

        # Iterate through all colors
        for _ in range(len(self.colors)):  # Subtract 1 to exclude the 'W' color
            if row_col == 1:
                # row
                row = random.randint(0, self.D - 1)
                color = self.choose_unused_color()
                self.used.append(color)
                self.grid[row] = [color] * self.D
                row_col = 0
            else:
                # col
                col = random.randint(0, self.D - 1)
                color = self.choose_unused_color()
                self.used.append(color)
                for x in range(self.D):
                    self.grid[x][col] = color
                row_col = 1
                    
    def choose_unused_color(self):
        while True:
            color = random.choice(self.colors)
            if color not in self.used:
                return color

    # This the random grid layout
    # PREDICTION: Not Dangerous / Dangerous
    # wrire to cut

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


if __name__ == "__main__":

    grid = Wire_Grid()
    grid.generate_Grid()
    grid.print_grid()
    