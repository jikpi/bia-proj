import random
import numpy as np


class City:
    def __init__(self, x: float, y: float, cid: int):
        self.x = x
        self.y = y
        self.id = cid

    def distance(self, other_city):
        # euklidovska vzdalenost mezi dvema mesty
        return np.sqrt((self.x - other_city.x) ** 2 + (self.y - other_city.y) ** 2)

    def __repr__(self):
        return f"City(id={self.id}, x={self.x:.2f}, y={self.y:.2f})"


class CityEntity:
    def __init__(self, city_list: list[City]):
        self.city_list = city_list
        self.cached_distance: float | None = None

    def distance(self):
        # vypocet celkove vzdalenosti cesty, od prvniho mesto do posledniho a zpatky do prvniho
        if self.cached_distance is not None:
            return self.cached_distance

        distance = 0
        for i in range(len(self.city_list)):
            distance += self.city_list[i].distance(self.city_list[(i + 1) % len(self.city_list)])

        self.cached_distance = distance
        return distance

    def crossover(self, other, rng=None):
        # nahodne vybrani bodu pro crossover
        if rng is None:
            rng = random.Random()

        # crossover bod (nebere v potaz prvni mesto)
        crossover_point = rng.randint(1, len(self.city_list) - 1)

        # nova entita s prvnim segmentem z prvniho rodice
        new_city_list = self.city_list[:crossover_point]

        # druhy segment z druheho rodice, zachova se poradi z druheho rodice
        for city in other.city_list:
            if city not in new_city_list:
                new_city_list.append(city)
                if len(new_city_list) == len(self.city_list):
                    break

        return CityEntity(new_city_list)

    def mutate(self, rng=None):
        # nahodna mutace prohozenim dvou mest, krome prvniho mesta
        if rng is None:
            rng = random.Random()

        idx_a = rng.randint(1, len(self.city_list) - 1)
        idx_b = rng.randint(1, len(self.city_list) - 1)
        self.city_list[idx_a], self.city_list[idx_b] = self.city_list[idx_b], self.city_list[idx_a]
        self.cached_distance = None


def generate_cities(num_cities=20, width=1000, height=1000, rng=None) -> list[City]:
    if rng is None:
        rng = random.Random()

    cities = []
    for i in range(num_cities):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        cities.append(City(x, y, i))

    return cities
