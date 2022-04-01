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
