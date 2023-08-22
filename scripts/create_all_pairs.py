import itertools
import pandas as pd
import csv
import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

train = pd.read_csv("./data/lotus_agg_train.csv.gz")
test = pd.read_csv("./data/lotus_agg_test.csv.gz")
df = pd.concat([train, test])

mol = df.structure_smiles_2D.unique()
species = df.organism_name.unique()

comb = itertools.product(species, mol)
with open("./data/all_pairs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["species", "molecule"])
    for s, m in comb:
        writer = csv.writer(file)
        writer.writerow([s, m])

print("Compressing the file with all pairs. This may take a while...")
command = "gzip ./data/all_pairs.csv"
subprocess.call(command, shell=True)
