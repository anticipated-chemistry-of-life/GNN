#!/usr/bin/env python
# coding: utf-8


import networkx as nx
import itertools
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from rdkit.Chem import AllChem, DataStructs

# Load data
df = pd.read_csv('230106_frozen_metadata.csv.gz')

#Remove duplicate organism-molecule pair
df_agg = df.groupby(['organism_taxonomy_09species',
                     'structure_smiles_2D']).size().reset_index(name='reference_wikidata')

df_agg = df.groupby(['organism_taxonomy_09species', 'structure_smiles_2D']).agg({
    'reference_wikidata': 'size',
    'organism_taxonomy_08genus': 'first',
    'organism_taxonomy_06family': 'first',
    'organism_taxonomy_05order': 'first',
    'organism_taxonomy_04class': 'first',
    'organism_taxonomy_03phylum': 'first',
    'organism_taxonomy_02kingdom': 'first',
    'organism_taxonomy_01domain': 'first',
    'structure_taxonomy_npclassifier_01pathway': 'first',
    'structure_taxonomy_npclassifier_02superclass': 'first',
    'structure_taxonomy_npclassifier_03class': 'first'
    # add other columns here as needed
}).reset_index()

df_agg['total_papers_molecule'] = df_agg.groupby(
    'structure_smiles_2D')['reference_wikidata'].transform('sum')
df_agg['total_papers_species'] = df_agg.groupby(
    'organism_taxonomy_09species')['reference_wikidata'].transform('sum')

#get random subset of the database (comment to have the full DB)
df_agg = df_agg.sample(n=20000).reset_index(drop=True)

# Fetch unique species and molecules and their respective features
unique_species_df = df_agg.drop_duplicates(subset=['organism_taxonomy_09species'])
unique_molecules_df = df_agg.drop_duplicates(subset=['structure_smiles_2D'])

# Fetch the corresponding features
species_features_df = unique_species_df[['organism_taxonomy_08genus', 'organism_taxonomy_06family', 
                                         'organism_taxonomy_05order', 'organism_taxonomy_04class', 
                                         'organism_taxonomy_03phylum', 'organism_taxonomy_02kingdom', 
                                         'organism_taxonomy_01domain']]
molecule_features_df = unique_molecules_df[['structure_taxonomy_npclassifier_01pathway', 
                                            'structure_taxonomy_npclassifier_02superclass', 
                                            'structure_taxonomy_npclassifier_03class']]

# Convert these dataframes to dummy/one-hot encoded dataframes
species_features_dummy = pd.get_dummies(species_features_df)
molecule_features_dummy = pd.get_dummies(molecule_features_df)
species_features_dummy.index = [i for i in unique_species_df['organism_taxonomy_09species']]
molecule_features_dummy.index = [i for i in unique_molecules_df['structure_smiles_2D']]

g = nx.Graph()
for i, row in df_agg.iterrows():
    g.add_edge(row['structure_smiles_2D'],
               row['organism_taxonomy_09species'],
              label="present_in")
    nx.set_node_attributes(g,
                           {row['structure_smiles_2D']: 'molecule',
                            row['organism_taxonomy_09species']: 'species'}, "label")
    nx.set_edge_attributes(g,
                          {(row['structure_smiles_2D'],
                            row['organism_taxonomy_09species']):
                           {'weight':row['reference_wikidata']}})

fps = [AllChem.MolFromSmiles(i) for i in unique_molecules_df['structure_smiles_2D']]
mols  = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in fps]

# Iterate over each pair of molecules
for i, j in itertools.combinations(range(len(fps)), 2):
    # Calculate the Tanimoto Similarity
    similarity = DataStructs.TanimotoSimilarity(mols[i], mols[j])
    # If the similarity is above 0.8, add an edge
    if similarity >= 0.8:
        g.add_edge(unique_molecules_df['structure_smiles_2D'].values[i], 
                  unique_molecules_df['structure_smiles_2D'].values[j],
                  label="similar_to")
        nx.set_edge_attributes(g,
                              {(unique_molecules_df['structure_smiles_2D'].values[i],
                              unique_molecules_df['structure_smiles_2D'].values[j]):{'weight': similarity}})

g = g.to_directed()
nx.write_graphml(g, "./lotus_DB_as_graph.gml")

molecule_features_dummy.to_csv("molecule_features_dummy.csv")
species_features_dummy.to_csv("species_features_dummy.csv")
df_agg.to_csv("lotus_aggregated.csv")

