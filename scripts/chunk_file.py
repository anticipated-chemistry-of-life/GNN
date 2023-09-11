import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

response = input(
    "This script will create a lot of files ! Make sure you have enough space on your computer (min 200G free) !\n\
Ideally run this on a cluster !\n\
Do you want to continue? (yes/no):"
)

if response.lower() != "yes":
    print("Aborting script execution.")
    sys.exit(0)  # Exit the script

train = pd.read_csv("./data/lotus_agg_train.csv.gz")
test = pd.read_csv("./data/lotus_agg_test.csv.gz")
df = pd.concat([train, test])

# Set chunk size (number of rows per chunk)
chunksize = len(
    df.structure_smiles_2D.unique()
)  # Modify this value based on your needs and available memory

# File path to the large data file
file_path = "./data/all_pairs.csv.gz"

# Iterate over chunks
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
    # Write each chunk to a separate parquet file
    chunk.to_parquet(f"./data/all_pairs_parquet/chunk_{i}.parquet")
