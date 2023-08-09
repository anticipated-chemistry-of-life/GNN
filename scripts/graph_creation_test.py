#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import pandas as pd

# Load data
df_test = pd.read_csv("./data/lotus_agg_test.csv.gz")

df_test.structure_smiles_2D.to_csv("./data/smiles_struct_test.csv")

g = nx.DiGraph()
for i, row in df_test.iterrows():
    g.add_edge(
        row["structure_smiles_2D"],
        row["organism_name"],
        label="present_in",
    )

    # create edge in oppsite direction
    g.add_edge(
        row["organism_name"],
        row["structure_smiles_2D"],
        label="has",
    )
    nx.set_node_attributes(
        g,
        {row["structure_smiles_2D"]: "molecule", row["organism_name"]: "species"},
        "label",
    )


nx.write_graphml(g, "./graph/test_graph.gml")
