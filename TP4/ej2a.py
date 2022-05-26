from models.Hopfield import Hopfield
from datasets.font_1 import alphabet


if __name__ == "__main__":
    dataset = list(alphabet.values())
    inputs = [alphabet["K"], alphabet["N"], alphabet["S"], alphabet["V"]]
    hopfield = Hopfield(inputs)
    k = [	[-1,-1,-1,+1,-1],
			[+1,-1,+1,-1,-1],
			[+1,+1,-1,-1,-1],
			[+1,-1,+1,-1,-1],
			[+1,-1,-1,+1,-1]]
    print(hopfield.associate(k))
