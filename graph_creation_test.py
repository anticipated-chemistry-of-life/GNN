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
df_test = pd.read_csv("./data/lotus_agg_test.csv.gz")

df_test.structure_smiles_2D.to_csv("./data/smiles_struct_test.csv")

g = nx.DiGraph()
for i, row in df_test.iterrows():
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


nx.write_graphml(g, "./graph/test_graph.gml")


