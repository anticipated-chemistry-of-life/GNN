import pandas as pd
import pygbif

# Load data
df = pd.read_csv('./data/230106_frozen_metadata.csv.gz', low_memory=False)
df = df.dropna(subset=['organism_name']).reset_index(drop=True)

#Remove duplicate organism-molecule pair
df_agg = df.groupby(['organism_name',
                     'structure_smiles_2D']).size().reset_index(name='reference_wikidata')

df_agg = df.groupby(['organism_name', 'structure_smiles_2D']).agg({
    'reference_wikidata': 'size',
    'organism_taxonomy_08genus': 'first',
    'organism_taxonomy_06family': 'first',
    'organism_taxonomy_05order': 'first',
    'organism_taxonomy_04class': 'first',
    'organism_taxonomy_03phylum': 'first',
    'organism_taxonomy_02kingdom': 'first',
    'organism_taxonomy_01domain': 'first',
    'structure_taxonomy_classyfire_01kingdom': 'first',
    'structure_taxonomy_classyfire_02superclass': 'first',
    'structure_taxonomy_classyfire_03class': 'first',
    'structure_taxonomy_classyfire_04directparent' : 'first'
    # add other columns here as needed
}).reset_index()

df_agg['total_papers_molecule'] = df_agg.groupby(
    'structure_smiles_2D')['reference_wikidata'].transform('sum')
df_agg['total_papers_species'] = df_agg.groupby(
    'organism_name')['reference_wikidata'].transform('sum')

#get taxonomy from GBIF
ls = []
for i in df_agg.organism_name.unique():
    ls.append(pygbif.species.name_backbone(name=i))
    
gbif = pd.DataFrame.from_records(ls)
gbif['organism_name'] = df_agg.organism_name.unique()
gbif.to_csv("./data_gbif/GBIF.csv.gz", compression="gzip")