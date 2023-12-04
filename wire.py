import math
from colorama import Back
import random


class Wire_Grid():
    def __init__(self) -> None:
        self.D = 20 # 20 x 20  Dimension of ship
        self.grid = []

        # colors possible in the ship, represented by indices, 0 - 3
        self.colors = ["RED", "BLUE", "YELLOW", "GREEN"]




    def generate_Grid(self) -> None:
        self.grid = [["-1" for _ in range(self.D)] for _ in range(self.D)]


        pass


if __name__ == "__main__":

    grid = Wire_Grid()
    grid.generate_Grid()
    for x in range(20):
        print(" ", end="")
        for y in range(20):
            print(grid.grid[x][y] + "  ", end="")
    