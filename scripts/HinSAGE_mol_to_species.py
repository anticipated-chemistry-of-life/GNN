#!/usr/bin/env python
# coding: utf-8

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import networkx as nx
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from tensorflow import keras
import tensorflow as tf

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_classification

import multiprocessing
import matplotlib.pyplot as plt


g = nx.read_graphml("./graph/train_graph.gml")
species_features_dummy = pd.read_csv("./data/species_features.csv.gz", index_col=0)
molecule_features_dummy = pd.read_csv(
    "./data/molecule_features.csv.gz", index_col=0
).astype("int8")
df_agg = pd.read_csv("./data/lotus_agg_train.csv.gz", index_col=0)


rdkit = pd.read_csv("./data/mol_dummy_rdkit.csv.gz", index_col=0).astype("int8")
molecule_features_dummy = molecule_features_dummy.merge(
    rdkit, left_index=True, right_index=True
)


species_test = species_features_dummy[
    ~species_features_dummy.index.isin(df_agg.organism_name)
].index
mol_test = molecule_features_dummy[
    ~molecule_features_dummy.index.isin(df_agg.structure_smiles_2D)
].index

species_feat = species_features_dummy[
    species_features_dummy.index.isin(df_agg.organism_name)
]
molecule_feat = molecule_features_dummy[
    molecule_features_dummy.index.isin(df_agg.structure_smiles_2D)
]

G = StellarGraph.from_networkx(
    g, node_features={"species": species_feat, "molecule": molecule_feat}
)
print(G.info())
G.check_graph_for_ml()


batch_size = 128  # default: 200
epochs = 30  # default: 20
num_samples = [3, 1]
num_workers = multiprocessing.cpu_count() - 2


# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.3 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=False, edge_label="present_in"
)


# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.3 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=False, edge_label="present_in"
)

print(G_train.info())
print(G_test.info())


train_gen = HinSAGELinkGenerator(
    G_train,
    batch_size=batch_size,
    num_samples=num_samples,
    head_node_types=["molecule", "species"],
    seed=42,
)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True, seed=42)


test_gen = HinSAGELinkGenerator(
    G_test,
    batch_size=batch_size,
    num_samples=num_samples,
    head_node_types=["molecule", "species"],
    seed=42,
)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test, seed=42)


hinsage_layer_sizes = [1024, 1024]
hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes,
    generator=train_gen,
    bias=True,
    dropout=0.3,
    activations=["elu", "selu"],
)


# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = hinsage.in_out_tensors()


prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="l1"
)(x_out)


model = keras.Model(inputs=x_inp, outputs=prediction)

initial_learning_rate = 0.1
final_learning_rate = 0.001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (
    1 / epochs
)
steps_per_epoch = int(edge_ids_train.shape[0] / batch_size)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch,
    decay_rate=learning_rate_decay_factor,
    staircase=True,
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.binary_crossentropy,
    metrics=["AUC"],
)


init_train_metrics = model.evaluate(train_flow, workers=num_workers, verbose=2)
init_test_metrics = model.evaluate(test_flow, workers=num_workers, verbose=2)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


callbacks = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, mode="auto", restore_best_weights=True
)

history = model.fit(
    train_flow,
    epochs=epochs,
    workers=num_workers,
    validation_data=test_flow,
    verbose=2,
    callbacks=[callbacks],
    validation_split=0.0,
    shuffle=True,
)


sg.utils.plot_history(history)


train_metrics = model.evaluate(train_flow, verbose=2)
test_metrics = model.evaluate(test_flow, verbose=2)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


def predict(model, flow, iterations=10):
    predictions = []
    for _ in range(iterations):
        predictions.append(model.predict(flow, workers=-1).flatten())

    return np.mean(predictions, axis=0)


test_pred = HinSAGELinkGenerator(
    G,
    batch_size=batch_size,
    num_samples=num_samples,
    head_node_types=["molecule", "species"],
    seed=42,
).flow(edge_ids_test, edge_labels_test, seed=42)


predictions = predict(model, test_pred)

model.save(f"./model/gbif_batch_{batch_size}_layer_{hinsage_layer_sizes[0]}_m_to_s")
