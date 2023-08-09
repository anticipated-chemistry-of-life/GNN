import pandas as pd

# Set chunk size (number of rows per chunk)
chunksize = 1000  # Modify this value based on your needs and available memory

# File path to the large data file
file_path = "./data/all_pairs.csv"

# Iterate over chunks
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
    # Write each chunk to a separate parquet file
    chunk.to_parquet(f"./data/out/chunk_{i}.parquet")
