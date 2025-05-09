import numpy as np


class LogisticMap:
    def __init__(self, iterations=1000,  # pocet iteraci funkce pro kazde 'a'
                 skip=200,  # kolik prvnich iteraci je odstraneno (pro "stabilni" vysledek)
                 x0=0.5):  # pocatecni hodnota x

        self.iterations = iterations
        self.skip = skip
        self.x0 = x0

    # iterace funkce pro parametr 'a'
    def iterate(self, a, x0=None):

        if x0 is None:
            x0 = self.x0

        x_values = np.zeros(self.iterations)
        x_values[0] = x0

        for i in range(1, self.iterations):
            # logisticka funkce
            x_values[i] = a * x_values[i - 1] * (1 - x_values[i - 1])

        # pole hodnot x
        return x_values

    def generate(self,
                 a_values  # pole parametru 'a'
                 ):
        results = []

        # pro vsechny parametry 'a'
        for a in a_values:
            x_values = self.iterate(a)

            # preskoceni prvnich 'skip' hodnot vysledku
            final_values = x_values[self.skip:]

            # pro kazde 'a' se vraci pole x hodnot
            results.append((a, final_values))

        # [(a, [x...]), ...]
        return results

    # vytvoreni datasetu
    def create_dataset(self, a_values,  # pole parametru 'a'
                       a_instances=100,  # kolik vysledku pro kazde 'a', s nahodnym x0
                       ):

        X = []
        y = []

        for a in a_values:
            # pro kazde 'a' vytvoreno 'a_instances' vysledku
            for _ in range(a_instances):
                # nahodne x0
                x0 = np.random.random()
                x_array = self.iterate(a, x0=x0)
                sample = x_array[self.skip:]

                X.append(sample)
                y.append(a)

        # ([[x1, x2, ...], [x1, x2, ...], ...], [a1, a2, ...])
        return np.array(X), np.array(y)
