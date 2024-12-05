import random

import numpy as np

from ex_4_ga_tsp.city_repr import generate_cities


class Ant:
    def __init__(self, initial_location: int):
        self.visited_cities = []
        self.distance = 0
        self.initial_location = initial_location


class AcoTspSolver:
    def __init__(self, city_count: int = 10, iteration_count=20, ant_count_ratio: float = 1, alpha: float = 1,
                 beta: float = 2,
                 evaporation_rate: float = 0.5,
                 city_seed: int = None,
                 x_size: int = 1000,
                 y_size: int = 1000):
        self.city_count = city_count
        self.ant_count = int(city_count * ant_count_ratio)
        self.alpha = alpha
        self.beta = beta
        self.iteration_count = iteration_count
        self.evaporation_rate = evaporation_rate
        self.best_ants = []
        self.best_ant = None
        self.x_size = x_size
        self.y_size = y_size
        self.city_seed = city_seed

        init_city_rng = random.Random(city_seed)
        cities = generate_cities(city_count, rng=init_city_rng, width=x_size, height=y_size)
        self.cities = cities

        self.city_distance_matrix = np.zeros((city_count, city_count))
        for i in range(city_count):
            for j in range(city_count):
                self.city_distance_matrix[i, j] = cities[i].distance(cities[j])

        self.pheromone_matrix = np.ones((city_count, city_count))

        # deep copy distance matrix mest
        city_distance_matrix_copy = np.copy(self.city_distance_matrix)

        # prevod na visibility matrix
        for i in range(city_count):
            for j in range(city_count):
                if i == j:
                    city_distance_matrix_copy[i, j] = 0
                else:
                    city_distance_matrix_copy[i, j] = 1 / city_distance_matrix_copy[i, j]

        self.visibility_matrix = city_distance_matrix_copy
        pass

    def search(self):
        iteration = 0

        while iteration < self.iteration_count:
            ants = []

            if self.ant_count > self.city_count:
                # nahodne se vybira startovni pozice pro kazdeho mravence
                for i in range(self.city_count):
                    # prvnich self.city_count mravencu zacina z kazdeho mesta
                    ants.append(Ant(i))

                # zbytek mravencu potom z nahodnych mest
                for i in range(self.city_count, self.ant_count):
                    ants.append(Ant(random.randint(0, self.city_count - 1)))
            else:
                # pokud je mravencu mene nez mest, kazdy mravenec zacina z nahodneho mesta
                for i in range(self.ant_count):
                    ants.append(Ant(random.randint(0, self.city_count - 1)))

            # loop pro mravence
            for i, ant in enumerate(ants):
                # sloupce v city matrix ktere mravenec navstivi, tedy 'false' na indexu znamena 'vynulovani' vzdalenosti
                # bool array ktera ma ze zacatku vsechno true
                unvisited_cities = np.ones(self.city_count, dtype=bool)
                # startovni pozice mravence je false
                unvisited_cities[ant.initial_location] = False

                ant.visited_cities.append(ant.initial_location)

                visited_city_count = 1

                # loop pro navstiveni mest
                while visited_city_count < self.city_count:
                    # dictionary {city, vzorec} pravdepodobnosti
                    probabilities = {}
                    for city in range(self.city_count):
                        if unvisited_cities[city]:
                            probabilities[city] = (self.pheromone_matrix[ant.visited_cities[-1], city] ** self.alpha) * \
                                                  (self.visibility_matrix[ant.visited_cities[-1], city] ** self.beta)

                    sum_visits = 0
                    for city in probabilities:
                        sum_visits += probabilities[city]

                    # seznam pravdepodobnosti pro navstiveni daneho mesta
                    probabilities = {city: probabilities[city] / sum_visits for city in probabilities}

                    # vyber dalsiho mesta pro mravence
                    # styl 'rulety'
                    random_val = random.random()
                    prob_sum = 0
                    selected_city = None
                    for city, prob in probabilities.items():
                        prob_sum += prob
                        if prob_sum >= random_val:
                            selected_city = city
                            break

                    ant.distance += self.city_distance_matrix[ant.visited_cities[-1], selected_city]
                    ant.visited_cities.append(selected_city)
                    unvisited_cities[selected_city] = False
                    visited_city_count += 1

                # pridani vzdalenosti z posledni mesto do prvniho mesta
                ant.distance += self.city_distance_matrix[ant.visited_cities[-1], ant.visited_cities[0]]
                ant.visited_cities.append(ant.visited_cities[0])

            # aktualizace pheromone matrixu

            # evaporace
            for i in range(self.city_count):
                for j in range(self.city_count):
                    self.pheromone_matrix[i, j] *= max(1 - self.evaporation_rate, 1e-4)

            # vlozeni pheromone z mravence do matice
            for ant in ants:
                p_amount = 1.0 / ant.distance
                for i in range(len(ant.visited_cities) - 1):
                    city1, city2 = ant.visited_cities[i], ant.visited_cities[i + 1]
                    self.pheromone_matrix[city1, city2] += p_amount
                    self.pheromone_matrix[city2, city1] += p_amount

            iteration += 1
            # ulozeni nejlepsiho mravence pro vizualizaci v teto iteraci
            best_ant = min(ants, key=lambda x: x.distance)
            if self.best_ant is None or best_ant.distance < self.best_ant.distance:
                self.best_ant = best_ant
            self.best_ants.append(best_ant)

            pass

        pass
