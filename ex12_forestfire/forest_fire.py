import numpy as np


class ForestFire:
    # stavy poli
    NONE = 0  # prazdne
    TREE = 1  # strom
    BURNING = 2  # horici strom
    BURNED = 3  # spaleny strom

    def __init__(self,
                 width=100,
                 height=100,
                 p=0.05,  # pravdepodobnost ze praznde pole / spaleny strom je nahrazeno stromem
                 f=0.001,  # pravdepodobnost, ze strom vzplane
                 density=0.5,  # hustota lesa
                 seed=None,
                 neighborhood="von_neumann",
                 start_burning=True
                 ):
        self.width = width
        self.height = height
        self.p = p
        self.f = f
        self.density = density
        self.neighborhood = neighborhood

        if seed is not None:
            np.random.seed(seed)

        self.grid = np.zeros((height, width), dtype=int)

        # podle hustoty je mrizka poli naplnena stromy
        mask = np.random.random((height, width)) < density
        self.grid[mask] = self.TREE

        # nastaveni nekolika nahodnych poli na horici strom
        if start_burning:
            num_initial_fires = max(1, int(width * height * 0.001))
            fire_y = np.random.randint(0, height, num_initial_fires)
            fire_x = np.random.randint(0, width, num_initial_fires)

            for y, x in zip(fire_y, fire_x):
                self.grid[y, x] = self.BURNING

    def step(self,
             p=None,
             f=None):

        if p is None:
            p = self.p
        if f is None:
            f = self.f

        # kopie mrizky pro zachovani predchoziho stavu
        new_grid = self.grid.copy()

        # prechody
        for y in range(self.height):
            for x in range(self.width):
                # pokud je pole prazdne nebo spalene, tak s pravd. 'p' zde vyroste strom
                if self.grid[y, x] == self.NONE or self.grid[y, x] == self.BURNED:
                    if np.random.random() < p:
                        new_grid[y, x] = self.TREE
                    else:
                        # spalene pole se meni na prazdne pokud zde nevyrostl strom
                        new_grid[y, x] = self.NONE

                # pokud je pole strom, zkontroluji se okolni pole zda jsou horici
                elif self.grid[y, x] == self.TREE:

                    # ulozeni indexu poli podle typu hledani
                    if self.neighborhood == "von_neumann":
                        # 4 okolni, "+"
                        neighbors = [(y - 1, x),
                                     (y, x + 1),
                                     (y + 1, x),
                                     (y, x - 1)]
                    else:
                        # moore, 8 okolnich, "+x"
                        neighbors = [(y - 1, x),
                                     (y - 1, x + 1),
                                     (y, x + 1),
                                     (y + 1, x + 1),
                                     (y + 1, x),
                                     (y + 1, x - 1),
                                     (y, x - 1),
                                     (y - 1, x - 1)]

                    # kontrola hranic okolnich souradnic, a zda sousedni pole hori
                    for neigb_y, neighb_x in neighbors:
                        if (0 <= neigb_y < self.height and 0 <= neighb_x < self.width
                                and self.grid[neigb_y, neighb_x] == self.BURNING):
                            # ma souseda co hori, strom vzplane
                            new_grid[y, x] = self.BURNING
                            break

                    # pravdepodobnost vplanuti f
                    if np.random.random() < f:
                        new_grid[y, x] = self.BURNING

                # strom, co hori, se stava spalenym
                elif self.grid[y, x] == self.BURNING:
                    new_grid[y, x] = self.BURNED

        self.grid = new_grid
