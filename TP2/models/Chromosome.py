class Chromosome:

    def __init__(self, genes):
        self.genes = genes

    def __str__(self):
        return str(self.genes)

    def __getitem__(self, items):
        return self.genes[items]

    def __len__(self):
        return len(self.genes)

    def __eq__(self, __o: object):
        return self.genes == __o

    def __str__(self):
        return str(self.genes)

    @property
    def W(self):
        return self.genes[:3]

    @property
    def w(self):
        return self.genes[3:9]

    @property
    def w_0(self):
        return self.genes[9:11]

    def mutate(self, mutation_method, mutation_rate):
        self.genes = mutation_method(self.genes, mutation_rate)
