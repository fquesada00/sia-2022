class Chromosome:

    def __init__(self, genes):
        self.genes = genes

    def __str__(self):
        return str(self.genes)

    def get_genes(self):
        return self.genes
