import itertools
import pandas as pd
import csv

df = pd.read_csv("./data/230106_frozen_metadata.csv.gz", low_memory=False)

mol = df.structure_smiles_2D.unique()[:10000]
species = df.organism_name.unique()[:1000]

comb = itertools.product(species, mol)
with open('./data/all_pairs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['species', 'molecule'])
    for s, m in comb:
        writer = csv.writer(file)
        writer.writerow([s, m])