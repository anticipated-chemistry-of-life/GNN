import pandas as pd
import sys
import os

# Add the parent directory to Python's module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.species import get_taxonomy


# Load data
df = pd.read_csv("./data/230106_frozen_metadata.csv.gz", low_memory=False)
df = df.dropna(subset=["organism_name"]).reset_index(drop=True)

# Remove duplicate organism-molecule pair
df_agg = (
    df.groupby(["organism_name", "structure_smiles_2D"])
    .size()
    .reset_index(name="reference_wikidata")
)

df_agg = (
    df.groupby(["organism_name", "structure_smiles_2D"])
    .agg(
        {
            "reference_wikidata": "size",
            "organism_taxonomy_08genus": "first",
            "organism_taxonomy_06family": "first",
            "organism_taxonomy_05order": "first",
            "organism_taxonomy_04class": "first",
            "organism_taxonomy_03phylum": "first",
            "organism_taxonomy_02kingdom": "first",
            "organism_taxonomy_01domain": "first",
            "structure_taxonomy_classyfire_01kingdom": "first",
            "structure_taxonomy_classyfire_02superclass": "first",
            "structure_taxonomy_classyfire_03class": "first",
            "structure_taxonomy_classyfire_04directparent": "first"
            # add other columns here as needed
        }
    )
    .reset_index()
)

df_agg["total_papers_molecule"] = df_agg.groupby("structure_smiles_2D")[
    "reference_wikidata"
].transform("sum")
df_agg["total_papers_species"] = df_agg.groupby("organism_name")[
    "reference_wikidata"
].transform("sum")

# get taxonomy from GBIF
gbif = get_taxonomy(df_agg.organism_name.unique(), n_cpus=4)
gbif.to_csv("./data/GBIF.csv.gz", compression="gzip")
