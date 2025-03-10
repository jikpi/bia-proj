import random
import sys

from BIA.ex_4_ga_tsp.city_repr import CityEntity, generate_cities


# reprezentuje jednu generaci
class TspGaGenerationData:
    def __init__(self, generation: int, entities: list[CityEntity]):
        self.generation: int = generation
        self.entities: list[CityEntity] = entities
        self.best_entity: CityEntity = min(entities, key=lambda x: x.distance())
        self.shortest_distance: float = self.best_entity.distance()

    def __repr__(self):
        return f"GenerationData(generation={self.generation}, best_entity={self.best_entity}, shortest_distance={self.shortest_distance:.2f})"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.entities
        elif isinstance(key, slice):
            return self.entities[:]
        raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.entities)


class GenAlgTspSolver:
    def __init__(self, x_size: int = 1000, y_size: int = 1000, np: int = 20, g: int = 200, d: int = 20,
                 initial_city_seed: int = None,
                 solution_seed: int = None,
                 mutation_prob: float = 0.1,
                 current_gen_selection_prob: float = 0.7):

        self.ent_bygen_list: list[TspGaGenerationData] = []
        self.np = np
        self.g = g
        self.d = d
        self.initial_city_seed = initial_city_seed
        self.solution_seed = solution_seed
        self.mutation_prob = mutation_prob
        self.current_gen_selection_prob = current_gen_selection_prob

        # master rng pro deterministicke generovani seedu do dalsich RNG
        self.master_rng = random.Random(solution_seed)

        # vygenerovani pocatecnich mest
        init_rng = random.Random(initial_city_seed)
        initial_cities = generate_cities(num_cities=self.d, rng=init_rng, height=x_size, width=y_size)

        # vytvoreni prvni generace nahodne
        first_gen = list[CityEntity]()
        for initial_entity in range(self.np):
            new_cities = initial_cities[:]
            shuffle_rng = random.Random(self.master_rng.randrange(sys.maxsize))
            # prvni mesto je zachovano
            first_city = new_cities[0]
            remaining_cities = new_cities[1:]
            shuffle_rng.shuffle(remaining_cities)
            new_cities = [first_city] + remaining_cities
            new_city_entity = CityEntity(new_cities)
            first_gen.append(new_city_entity)

        self.ent_bygen_list.append(TspGaGenerationData(0, first_gen))

    def search(self):
        for gen in range(self.g):
            # posledni generace
            current_gen = self.ent_bygen_list[-1][:]
            # nova generace, zacina jako kopie posledni
            new_gen = self.ent_bygen_list[-1][:]

            # pro kazdou entitu v posledni generaci
            for i in range(self.np):
                parent_a = current_gen[i]

                selection_rng = random.Random(self.master_rng.randrange(sys.maxsize))
                parent_choice_rng = random.Random(self.master_rng.randrange(sys.maxsize))
                mutation_chance_rng = random.Random(self.master_rng.randrange(sys.maxsize))
                crossover_rng = random.Random(self.master_rng.randrange(sys.maxsize))
                mutation_rng = random.Random(self.master_rng.randrange(sys.maxsize))

                # 70% sance vybrat rodice B z aktualni generace, 30% sance z nove
                if selection_rng.random() < self.current_gen_selection_prob:
                    parent_b = parent_choice_rng.choice(current_gen)
                else:
                    parent_b = parent_choice_rng.choice(new_gen)

                if parent_a is parent_b:
                    while parent_a is parent_b:
                        parent_b = parent_choice_rng.choice(current_gen)

                # vytvoreni potomka pomoci crossover
                offspring_ab = parent_a.crossover(parent_b, rng=crossover_rng)

                # 10% sance na mutaci potomka
                if mutation_chance_rng.random() < self.mutation_prob:
                    offspring_ab.mutate(rng=mutation_rng)

                if offspring_ab.distance() < parent_a.distance():
                    new_gen[i] = offspring_ab

            self.ent_bygen_list.append(TspGaGenerationData(gen + 1, new_gen))
