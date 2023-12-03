import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import networkx as nx
import tensorflow as tf
import numpy as np
from utils.utils import predict
import matplotlib.pyplot as plt
import seaborn as sns


def user_query(lotus_agg: pd.DataFrame, species_features, molecule_features, query):
    if query in set(lotus_agg["structure_smiles_2D"]):
        return pd.DataFrame(
            {
                "molecule": query,
                "species": list(
                    set(species_features.index)
                    - set(
                        lotus_agg[lotus_agg.structure_smiles_2D == query].organism_name
                    )
                ),
            }
        )
    elif query in set(lotus_agg["organism_name"]):
        return pd.DataFrame(
            {
                "species": query,
                "molecule": list(
                    set(molecule_features.index)
                    - set(
                        lotus_agg[lotus_agg.organism_name == query].structure_smiles_2D
                    )
                ),
            }
        )
    else:
        raise ValueError(
            f"The query you gave : {query} is not in LOTUS ! Please input only SMILES or species that are present in LOTUS."
        )


def main():
    parser = argparse.ArgumentParser(description="LOTUS Prediction Script")
    parser.add_argument(
        "--query", type=str, required=True, help="Input molecule or species query"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Top N values to return (default: -1 returns entire dataframe)",
    )

    args = parser.parse_args()
    # import Keras models
    print("Loading trained models")
    model_m_to_s = tf.keras.models.load_model(
        "./model/gbif_batch_128_layer_1024_m_to_s", compile=True
    )
    model_s_to_m = tf.keras.models.load_model(
        "./model/gbif_batch_128_layer_1024_s_to_m", compile=True
    )

    # Import features
    print("Importing features")
    species_features = pd.read_csv("./data/species_features.csv.gz", index_col=0)
    molecule_features = pd.read_csv("./data/molecule_features.csv.gz", index_col=0)

    df_agg = pd.read_csv("./data/lotus_agg_test.csv.gz", index_col=0)
    df_agg_train = pd.read_csv("./data/lotus_agg_train.csv.gz", index_col=0)
    df = pd.concat([df_agg, df_agg_train])
    del df_agg
    del df_agg_train

    rdkit = pd.read_csv("./data/mol_dummy_rdkit.csv.gz", index_col=0).astype("uint8")
    rdkit.columns = rdkit.columns.astype(str)

    print("Importing LOTUS as graph")
    g_train = nx.read_graphml("./graph/train_graph.gml")
    g_test = nx.read_graphml("./graph/test_graph.gml")
    g_lotus = nx.compose(g_train, g_test)
    del g_train
    del g_test

    data = user_query(df, species_features, molecule_features, query=args.query)

    out = predict(
        g_lotus,
        model_m_to_s,
        model_s_to_m,
        data,
        molecule_features,
        rdkit,
        species_features,
        n_cpus=4,
    )

    out = out.sort_values(by="prob", ascending=False)
    if args.n != -1:
        out = out.head(args.n)
    return out


if __name__ == "__main__":
    result = main()
    print(result)
