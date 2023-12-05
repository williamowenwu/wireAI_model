import math
from colorama import Back
import random


class Wire_Grid():
    def __init__(self) -> None:
        self.D = 20 # 20 x 20  Dimension of ship
        self.grid = []

        # colors possible in the grid
        self.colors = ["R", "B", "Y", "G", "W"]

    def generate_Grid(self) -> None:
        self.grid = [["W" for _ in range(self.D)] for _ in range(self.D)]
        
        row_col = random.randint(0,19)
        


    # This the random grid layout
    # PREDICTION: Not Dangerous / Dangerous
    # wrire to cut

    def print_grid(self) -> None:
        for x in range(grid.D):
            print(" ", end="")
            for y in range(grid.D):
                if grid.grid[x][y] == "W": 
                    print(Back.WHITE + " ", end = " ")     
            print(Back.RESET) 


if __name__ == "__main__":

    grid = Wire_Grid()
    grid.generate_Grid()
    grid.print_grid()
    