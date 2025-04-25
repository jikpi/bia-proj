import numpy as np


class Terrain:
    def __init__(self, pt_init=1.0,  # pocatecni hodnota pt (perturbace, mira zmeny vysky)
                 pt_damping=0.5,  # pt je vynasobeno timto cislem kazdy step, napr 0.1 -> rychle zmenseni pt
                 seed=None):
        self.pt_amount = pt_init
        self.pt_damping = pt_damping
        self.current_pt_amount = pt_init
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # pocatkem je jeden quad (4 vrcholy), v matici jsou ulozeny vrcholy
        self.size = 2
        self.heights = np.zeros((self.size, self.size))

        # pocatecni vysky 4 vrcholu, napr. nahodne
        self.heights[0, 0] = np.random.uniform(-0.5, 0.5) * self.pt_amount
        self.heights[0, 1] = np.random.uniform(-0.5, 0.5) * self.pt_amount
        self.heights[1, 0] = np.random.uniform(-0.5, 0.5) * self.pt_amount
        self.heights[1, 1] = np.random.uniform(-0.5, 0.5) * self.pt_amount

    def step(self):
        # nova matice s 2x vrcholy
        new_size = self.size * 2
        new_heights = np.zeros((new_size, new_size))

        # zkopirovani predchozich hodnot na 2 * index (kazdy druhy index v nove matici)
        for i in range(self.size):
            for j in range(self.size):
                new_heights[i * 2, j * 2] = self.heights[i, j]

        # diamond-square algoritmus pro vyplneni novych vrcholu

        # nastaveni novych vrcholu v radach mezi starymi vrcholy
        for i in range(0, new_size, 2):  # sude rady (stare vrcholy)
            for j in range(1, new_size - 1, 2):  # sude sloupce (nove vrcholy)
                # zkopirovani a zprumerovani hodnoty sousednich vrcholu vlevo a vpravo
                new_heights[i, j] = (new_heights[i, j - 1] + new_heights[i, j + 1]) / 2.0
                # perturbace noveho vrcholu
                new_heights[i, j] += np.random.uniform(-0.5, 0.5) * self.current_pt_amount

        # nastaveni novych vrcholu v sloupcich mezi starymi vrcholy
        for i in range(1, new_size - 1, 2):  # sude radky (nove vrcholy)
            for j in range(0, new_size, 2):  # sude sloupce (stare vrcholy)
                # zkopirovani a zprumerovani hodnoty sousednich vrcholu nahore a dole
                new_heights[i, j] = (new_heights[i - 1, j] + new_heights[i + 1, j]) / 2.0
                # perturbace noveho vrcholu
                new_heights[i, j] += np.random.uniform(-0.5, 0.5) * self.current_pt_amount

        # nastaveni novych vrcholu diagonalne dolu doprava od starych vrcholu
        for i in range(1, new_size - 1, 2):  # sude radky (nove vrcholy)
            for j in range(1, new_size - 1, 2):  # sude sloupce (nove vrcholy)
                # prumer ze 4 sousednich vrcholu
                new_heights[i, j] = (
                                            new_heights[i - 1, j] +  # nahoru
                                            new_heights[i + 1, j] +  # dolu
                                            new_heights[i, j - 1] +  # doleva
                                            new_heights[i, j + 1]  # doprava
                                    ) / 4.0
                # perturbace noveho vrcholu
                new_heights[i, j] += np.random.uniform(-0.5, 0.5) * self.current_pt_amount

        self.heights = new_heights
        self.size = new_size

        # vynasobeni pt
        self.current_pt_amount *= self.pt_damping

        return self.heights

    # nastaveni "hladiny more"
    def setSea(self, min_height):
        self.heights = np.maximum(self.heights, min_height)
        return self.heights
