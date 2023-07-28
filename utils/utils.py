import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import networkx as nx
import itertools
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from tensorflow import keras
import tensorflow as tf
import category_encoders as ce
import stellargraph
from stellargraph.mapper import HinSAGELinkGenerator
from tensorflow import keras

def load_input_data(path):
    data = pd.read_csv(path, index_col=0)
    
    if len(data.columns) != 2:
        raise IndexError("Input data must have 2 columns !")
    if data.columns.isnull().any():
        raise ValueError("The input DataFrame has unnamed columns ! Columns should be named.")
    
    return data

def check_which_model_to_use(G: nx.DiGraph, data: pd.DataFrame):
    if len(data.columns) != 2:
        raise IndexError("Input data must have 2 columns !")

    if data.columns[0] != 'molecule' and data.columns[0] != 'species':
        raise ValueError("First column must be named either : 'molecule' or 'species' !")

    if data.columns[1] != 'molecule' and data.columns[1] != 'species':
        raise ValueError("Second column must be named either : 'molecule' or 'species' ! ")

    # Get set of nodes in the graph for fast lookup
    nodes = set(G.nodes())
    data_copy = data.copy()
    
    # Create new column 'model' based on whether 'molecule' and 'species' are in nodes
    data_copy['model'] = data_copy.apply(lambda row: _get_model(row, nodes), axis=1)

    # Get numpy arrays
    if data.columns[0] == 'molecule':
        m_to_s = data_copy[data_copy['model'] == 'm_to_s'][['molecule', 'species']].values
        s_to_m = data_copy[data_copy['model'] == 's_to_m'][['molecule', 'species']].values
        both_unknown = data_copy[data_copy['model'] == 'both_unknown'][['molecule', 'species']].values
        both_known = data_copy[data_copy['model'] == 'both_known'][['molecule', 'species']].values
        
    else :
        m_to_s = data_copy[data_copy['model'] == 'm_to_s'][['species', 'molecule']].values
        s_to_m = data_copy[data_copy['model'] == 's_to_m'][['species', 'molecule']].values
        both_unknown = data_copy[data_copy['model'] == 'both_unknown'][['species', 'molecule']].values
        both_known = data_copy[data_copy['model'] == 'both_known'][['species', 'molecule']].values

    return data_copy, m_to_s, s_to_m, both_unknown, both_known


def _get_model(row, nodes):
    if (row['molecule'] not in nodes) and (row['species'] not in nodes):
        return 'both_unknown'
    elif (row['molecule'] not in nodes) and (row['species'] in nodes):
        return 'm_to_s'
    elif (row['molecule'] in nodes) and (row['species'] not in nodes):
        return 's_to_m'
    else:
        return 'both_known'

def add_nodes_to_graph(G: nx.DiGraph, data: pd.DataFrame)-> nx.DiGraph :
    ''' This function will add the missing node to the graph before prediction. Input should be a pandas dataframe
    with one column with the species, and the other with the molecules that will be predicted. There must be a
    column name to the input data. 
    '''
    G_copy = G.copy()
    assert len(data.columns) == 3, "Input data must have 3 columns ! Mol, Species, and Model. Something went wrong in the previous steps"
    if data.columns.isnull().any():
        raise ValueError("The input DataFrame has unnamed columns ! Columns should be named.")
    
    if data.columns[0] != 'molecule' and data.columns[0] != 'species':
        raise ValueError("First column must be named either : 'molecule' or 'species' !")
        
    if data.columns[1] != 'molecule' and data.columns[1] != 'species':
        raise ValueError("Second column must be named either : 'molecule' or 'species' ! ")
    
    first_column_set = set(data.iloc[:, 0])
    second_column_set = set(data.iloc[:, 1])
    
    for name in first_column_set:
        if name not in G_copy:
            G_copy.add_node(name, label=data.columns[0])
            
    for name in second_column_set:
        if name not in G_copy:
            G_copy.add_node(name, label=data.columns[1])
    
    return G_copy


def nx_to_stellargraph(g: nx.DiGraph,
                       molecule_features: pd.core.frame.DataFrame,
                       species_features: pd.core.frame.DataFrame
                       ) -> stellargraph.core.graph.StellarDiGraph:
    
    if set(g.nodes())-set(molecule_features.index)-set(species_features.index) != set():
        raise Exception("Some nodes do not have features ! Please check your graph or your features.")
        
    mol_feat = molecule_features[molecule_features.index.isin(g.nodes())]
    species_feat = species_features[species_features.index.isin(g.nodes())]
    G = StellarGraph.from_networkx(
        g,
        node_features={'species': species_feat,
                       'molecule': mol_feat}
        )
    
    G.check_graph_for_ml()
    print(G.info())
    
    return G


def create_flow(graph: stellargraph.core.graph.StellarDiGraph,
                data: pd.DataFrame,
                array: np.ndarray,
                unknown_node = 'molecule'
                ):
    batch_size = []
    #if the first column is the one we want to predict, keep head node types as they are. 
    if data.columns[0] == unknown_node:
        flow = HinSAGELinkGenerator(
            graph,
            batch_size=1024,
            num_samples=[3,1],
            head_node_types=[data.columns[0], data.columns[1]]).flow(array,
                                                                     np.ones(len(array)).reshape(-1,1))
    
    # else switch the head node types and revert the columns of the array so that they match the 
    # head node types order
    else:
        flow = HinSAGELinkGenerator(
        graph,
        batch_size=1024,
        num_samples=[3,1],
        head_node_types=[data.columns[1], data.columns[0]]).flow(array[:, ::-1],
                                                                 np.ones(len(array)).reshape(-1,1))
        
    return flow

def _predict(model, flow, iterations=7):
    predictions = []
    for _ in range(iterations):
        predictions.append(model.predict(flow, workers=-1).flatten())

    return np.mean(predictions, axis=0)

def _predict_using_both_models(model_m_to_s,
                              model_s_to_m,
                              flow_m: stellargraph.mapper.sequences.LinkSequence,
                              flow_s: stellargraph.mapper.sequences.LinkSequence) -> np.ndarray:
    
    #do prediction both ways and average them. 
    print("Predict both : running molecule to species predictions...")
    a = _predict(model_m_to_s, flow_m)
    
    print("Predict both : running species to molecules prediction...")
    b = _predict(model_s_to_m, flow_s)
    
    assert len(a)==len(b), f"Forward is of length {len(a)} and backward is of length {len(b)}. They should be the same."
    
    return (a+b)/2


def predict(graph : nx.DiGraph,
            model_m_to_s,
            model_s_to_m,
            data: pd.DataFrame,
            molecule_features: pd.DataFrame,
            species_features: pd.DataFrame
            ):
    
    #first check which model should be used for each row
    print("Checking which model should be used for each row...")
    data_out, m_to_s, s_to_m, both_unknown, both_known = check_which_model_to_use(graph, data)
    
    #the add the missing nodes to the graph
    print("Adding missing nodes to the graph...")
    graph_with_nodes = add_nodes_to_graph(graph, data_out)
    
    #convert NetworkX graph to Stellargraph
    print("Converting NetworkX to Stellargraph...")
    graph_stellar = nx_to_stellargraph(graph_with_nodes, molecule_features, species_features)
    
    del graph_with_nodes
    
    # Initialize outputs as empty lists
    out_m = np.array([])
    out_s = np.array([])
    out_both_unknown = np.array([])
    out_both_known = np.array([])
    
    #create the different flows according to what we want to do
    if m_to_s.size != 0 :
        print("Creating mol to species flow...")
        flow_m = create_flow(graph_stellar, data, m_to_s, 'molecule')
        
        print("Predicting mol to species...")
        out_m = _predict(model_m_to_s, flow_m)
        del flow_m
        
    if s_to_m.size != 0:
        print("Creating species to mol flow...")
        flow_s = create_flow(graph_stellar, data, s_to_m, 'species')
        
        print("Predicting species to mol...")
        out_s = _predict(model_s_to_m, flow_s)
        del flow_s

    if both_unknown.size != 0:
        print("Creating 'forward', 'backward' flow for UNKNOWN molecule AND species...")
        flow_m_both_unknown = create_flow(graph_stellar, data, both_unknown, 'molecule')
        flow_s_both_unknown = create_flow(graph_stellar, data, both_unknown, 'species')
        out_both_unknown = _predict_using_both_models(model_m_to_s, model_s_to_m, flow_m_both_unknown, flow_s_both_unknown)
        del flow_m_both_unknown
        del flow_s_both_unknown
        
    if both_known.size != 0:
        print("Creating 'forward', 'backward' flow for KNOWN molecule AND species...")
        flow_m_both_known = create_flow(graph_stellar, data, both_known, 'molecule')
        flow_s_both_known = create_flow(graph_stellar, data, both_known, 'species')
        out_both_known = _predict_using_both_models(model_m_to_s, model_s_to_m, flow_m_both_known, flow_s_both_known)
        del flow_m_both_known
        del flow_s_both_known
    
    if data.columns[0] == 'molecule':
        out_df = pd.DataFrame(np.vstack((m_to_s, s_to_m, both_unknown, both_known)),
                              columns=['molecule', 'species'])
    else:
        out_df = pd.DataFrame(np.vstack((m_to_s, s_to_m, both_unknown, both_known)),
                              columns=['species', 'molecule'])
        
    out_df['prob'] = np.concatenate((out_m, out_s, out_both_unknown, out_both_known))
        
    
    return pd.merge(data_out, out_df, on=['species', 'molecule'])