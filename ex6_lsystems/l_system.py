class LSystem:
    def __init__(self, axiom,
                 rules,  # pravidla (Neterminal -> pravidlo), Muze byt vice (musi byt velke pismeno)
                 angle):
        self.axiom = axiom
        self.rules = {node: replacement for node, replacement in rules}
        if not all(node.isupper() and len(node) == 1 for node in self.rules.keys()):
            raise ValueError("Node rule must be uppercase")

        self.angle = angle
        self.current = axiom

    def expand(self, iterations=1):
        # aplikace pravidel na momentalni stav
        for _ in range(iterations):
            result = []
            for char in self.current:
                # pouziti pravidla, pokud se neterminal vyskytuje v pravidlech
                result.append(self.rules.get(char, char))

            # ulozeni noveho stavu
            self.current = ''.join(result)

        return self

    def get_string(self):
        return self.current
