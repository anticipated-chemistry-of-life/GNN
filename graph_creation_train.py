#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import itertools
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from rdkit.Chem import AllChem, DataStructs
import category_encoders as ce

# Load data
df = pd.read_csv('./data/230106_frozen_metadata.csv.gz', low_memory=False)
df = df.dropna(subset=['organism_name']).reset_index(drop=True)

#Remove duplicate organism-molecule pair
df_agg = df.groupby(['organism_name',
                     'structure_smiles_2D']).size().reset_index(name='reference_wikidata')

df_agg = df.groupby(['organism_name', 'structure_smiles_2D']).agg({
    'reference_wikidata': 'size',
    'organism_taxonomy_08genus': 'first',
    'organism_taxonomy_07tribe': 'first',
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

#get random subset of the database (comment to have the full DB)
#df_agg_train = df_agg_train.sample(n=100000).reset_index(drop=True)

# Fetch unique species and molecules and their respective features
unique_species_df = df_agg.drop_duplicates(subset=['organism_name'])
unique_molecules_df = df_agg.drop_duplicates(subset=['structure_smiles_2D'])

# Fetch the corresponding features
species_features_df = unique_species_df[['organism_taxonomy_01domain',
                                         'organism_taxonomy_02kingdom',
                                         'organism_taxonomy_03phylum',
                                         'organism_taxonomy_04class',
                                         'organism_taxonomy_05order',
                                         'organism_taxonomy_06family',
                                         'organism_taxonomy_07tribe',
                                         'organism_taxonomy_08genus',
                                         'organism_name']]

molecule_features_df = unique_molecules_df[['structure_taxonomy_classyfire_01kingdom',
                                            'structure_taxonomy_classyfire_02superclass',
                                            'structure_taxonomy_classyfire_03class',
                                            'structure_taxonomy_classyfire_04directparent']]


# create features for species
encoder_species = ce.BinaryEncoder(cols=[col for col in species_features_df.columns])
species_features_dummy = encoder_species.fit_transform(species_features_df)


encoder_molecule = ce.BinaryEncoder(cols=[col for col in molecule_features_df.columns])
molecule_features_dummy = encoder_molecule.fit_transform(molecule_features_df)

species_features_dummy.index = [i for i in unique_species_df['organism_name']]
molecule_features_dummy.index = [i for i in unique_molecules_df['structure_smiles_2D']]

sample_fraction = 0.2  # 20% for example
df_test = df_agg.sample(frac=sample_fraction, random_state=42)
df_test.to_csv("./data/lotus_agg_test.csv.gz", compression="gzip")
df_train = df_agg.drop(df_test.index)
df_train.structure_smiles_2D.to_csv("./data/smiles_struct_train.csv")

g = nx.DiGraph()
for i, row in df_train.iterrows():
    g.add_edge(row['structure_smiles_2D'],
               row['organism_name'],
              label="present_in")

    #create edge in oppsite direction
    g.add_edge(row['organism_name'],
               row['structure_smiles_2D'],
              label="has")
    nx.set_node_attributes(g,
                           {row['structure_smiles_2D']: 'molecule',
                            row['organism_name']: 'species'},
                           "label")
    #nx.set_edge_attributes(g,
    #                      {(row['structure_smiles_2D'],
    #                        row['organism_name']):
    #                       {'weight':
    #                           1-np.exp(-0.005 * row['total_papers_species']- 0.005*row['total_papers_molecule'])}})
    #nx.set_edge_attributes(g,
    #                      {(row['organism_name'],
    #                       row['structure_smiles_2D']):
    #                       {'weight':
    #                           1-np.exp(-0.005 * row['total_papers_species']- 0.005*row['total_papers_molecule'])}})


fps = [AllChem.MolFromSmiles(i) for i in unique_molecules_df['structure_smiles_2D']]
mols  = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in fps]
mol_dum = [np.array(i) for i in mols]
mol_dum = pd.DataFrame(mol_dum)

# Iterate over each pair of molecules
#for i, j in itertools.combinations(range(len(fps)), 2):
#    # Calculate the Tanimoto Similarity
#    similarity = DataStructs.TanimotoSimilarity(mols[i], mols[j])
#    # If the similarity is above 0.9, add an edge
#    if similarity >= 0.8:
#        g.add_edge(unique_molecules_df['structure_smiles_2D'].values[i], 
#                  unique_molecules_df['structure_smiles_2D'].values[j],
#                  label="similar_to")
#        g.add_edge(unique_molecules_df['structure_smiles_2D'].values[j], 
#                  unique_molecules_df['structure_smiles_2D'].values[i],
#                  label="similar_to")
#        nx.set_edge_attributes(g,
#                              {(unique_molecules_df['structure_smiles_2D'].values[i],
#                              unique_molecules_df['structure_smiles_2D'].values[j]):{'weight': similarity}})
#        nx.set_edge_attributes(g,
#                              {(unique_molecules_df['structure_smiles_2D'].values[j],
#                              unique_molecules_df['structure_smiles_2D'].values[i]):{'weight': similarity}})


mol_dum.index = [i for i in unique_molecules_df['structure_smiles_2D']]


nx.write_graphml(g, "./graph/train_graph.gml")

molecule_features_dummy.to_csv("./data/molecule_features.csv.gz", compression="gzip")
species_features_dummy.to_csv("./data/species_features.csv.gz", compression="gzip")
df_train.to_csv("./data/lotus_agg_train.csv.gz", compression="gzip")
mol_dum.to_csv("./data/mol_dummy_rdkit.csv.gz", compression="gzip")