#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pandas as pd
import numpy as np
import dgl
import torch
from rdkit.Chem import AllChem, DataStructs

# Load data
df = pd.read_csv('230106_frozen_metadata.csv.gz')

# Preprocess the data as in your original script...

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

#get the first 30'000 entries (remove for full dataset)
df_agg = df_agg.iloc[:10000, :]


# In[2]:


# create two dataframes for nodes
df_species = pd.DataFrame(df_agg['organism_taxonomy_09species'].unique(), columns=['species'])
df_molecule = pd.DataFrame(df_agg['structure_smiles_2D'].unique(), columns=['molecule'])

# assign unique integer id for nodes
df_species['species_id'] = np.arange(len(df_species))
df_molecule['molecule_id'] = np.arange(len(df_molecule))

# map species and molecule to their ids in df_agg
df_agg = pd.merge(df_agg, df_species, left_on='organism_taxonomy_09species', right_on='species', how='left')
df_agg = pd.merge(df_agg, df_molecule, left_on='structure_smiles_2D', right_on='molecule', how='left')

# Create edge weights based on the formula: 1 - np.exp(-0.5 * reference_wikidata)
df_agg['edge_weight'] = 1 - np.exp(-0.5 * df_agg['reference_wikidata'])

# Convert edge weights to a tensor
edge_weights = torch.from_numpy(df_agg['edge_weight'].values.astype(np.float32))


# In[3]:


# create edge list
edges_s_to_m = df_agg[['species_id', 'molecule_id']].values.T
edges_m_to_s = df_agg[['molecule_id', 'species_id']].values.T


# In[4]:


# create the heterograph
g = dgl.heterograph({
    ('species', 'has', 'molecule'): (edges_s_to_m[0], edges_s_to_m[1]),
    ('molecule', 'is_present_in', 'species'): (edges_m_to_s[0], edges_m_to_s[1]),
    ('molecule', 'similar_to', 'molecule'): ([], [])
})


# In[5]:


# Add edge weights to the graph
g.edges['has'].data['weight'] = edge_weights
g.edges['is_present_in'].data['weight'] = edge_weights


# In[6]:


# Calculate molecular fingerprints
#fps = df_agg['structure_smiles_2D'].unique()
fps = [AllChem.MolFromSmiles(i) for i in df_molecule.molecule]
mols  = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in fps]

# Create a list to store the edges and their weights
similar_edges = []
similar_edge_weights = []

# Iterate over each pair of molecules
for i, j in itertools.combinations(range(len(fps)), 2):
    # Calculate the Tanimoto Similarity
    similarity = DataStructs.TanimotoSimilarity(mols[i], mols[j])
    # If the similarity is above 0.7, add an edge
    if similarity >= 0.7:
        # Add both (i, j) and (j, i) edges to the list
        similar_edges.extend([(i, j), (j, i)])
        # Add the similarity as the edge weight for both edges
        similar_edge_weights.extend([similarity, similarity])

# Convert the lists to tensors
similar_edges = torch.tensor(similar_edges, dtype=torch.int64)
similar_edge_weights = torch.tensor(similar_edge_weights, dtype=torch.float32)

# Add the 'similar_to' edges to the graph
g.add_edges(similar_edges[:, 0], similar_edges[:, 1], etype='similar_to')
g.edges['similar_to'].data['weight'] = similar_edge_weights


# In[7]:


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

# Add these features to the corresponding nodes in the graph
g.nodes['species'].data.update(
    {key: torch.from_numpy(species_features_dummy[key].values) for key in species_features_dummy}
)
g.nodes['molecule'].data.update(
    {key: torch.from_numpy(molecule_features_dummy[key].values) for key in molecule_features_dummy}
)


# In[9]:


# Finally, save the heterograph
from dgl.data.utils import save_graphs
save_graphs("./hetero_graph.bin", [g])

