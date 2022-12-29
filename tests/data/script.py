from tqdm import tqdm
import random

with open("edges.txt") as reader:
    with open("new.txt", "x") as writer:
        lines = reader.readlines()
        for line in tqdm(lines):
            writer.write(f"{line[:-1]},{random.randint(0,1)}\n")

