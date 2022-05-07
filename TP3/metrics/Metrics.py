class Metrics:
    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __iter__(self):
        yield self.accuracy
        yield self.precision
        yield self.recall
        yield self.f1

    def __str__(self):
        return f"{self.accuracy:.5f}\t{self.precision:.5f}\t{self.recall:.5f}\t{self.f1:.5f}"
