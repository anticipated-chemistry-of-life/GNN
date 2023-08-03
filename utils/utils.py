import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import networkx as nx
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from tensorflow import keras
import tensorflow as tf
import category_encoders as ce
import stellargraph
from stellargraph.mapper import HinSAGELinkGenerator
from tensorflow import keras
from utils.molecules import inchikey_to_smiles, smiles_to_classyfire, smiles_to_fingerprint
from utils.species import get_taxonomy
from utils.encoding import binary_encode_df

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

    valid_names = ['molecule', 'species']
    if data.columns[0] not in valid_names:
        raise ValueError("First column must be named either : 'molecule' or 'species' !")
    if data.columns[1] not in valid_names:
        raise ValueError("Second column must be named either : 'molecule' or 'species' !")

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
    if len(data.columns) != 3:
        raise IndexError("Input data must have 3 columns ! Mol, Species, and Model. Something went wrong in the previous steps")
    
    if data.columns.isnull().any():
        raise ValueError("The input DataFrame has unnamed columns ! Columns should be named.")
    
    valid_names = ['molecule', 'species']
    if data.columns[0] not in valid_names:
        raise ValueError("First column must be named either : 'molecule' or 'species' !")
    if data.columns[1] not in valid_names:
        raise ValueError("Second column must be named either : 'molecule' or 'species' !")
    
    first_column_set = set(data['molecule'])
    second_column_set = set(data['species'])
    
    for name in first_column_set:
        if name not in G_copy:
            G_copy.add_node(name, label='molecule')
            
    for name in second_column_set:
        if name not in G_copy:
            G_copy.add_node(name, label='species')
    
    return G_copy


def nx_to_stellargraph(g: nx.DiGraph,
                       molecule_features: pd.DataFrame,
                       species_features: pd.DataFrame
                       ) -> stellargraph.core.graph.StellarDiGraph:
    
    if set(g.nodes())-set(molecule_features.index)-set(species_features.index) != set():
        print(set(g.nodes())-set(molecule_features.index)-set(species_features.index))
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
    
    if len(a)!=len(b):
        raise IndexError(f"Forward is of length {len(a)} and backward is of length {len(b)}. They should be the same.")
    
    return (a+b)/2

def convert_to_smiles(data: pd.DataFrame, n_cpus=4):
    valid_names = ['molecule', 'species']
    if data.columns[0] not in valid_names:
        raise ValueError("First column must be named either : 'molecule' or 'species' !")
    if data.columns[1] not in valid_names:
        raise ValueError("Second column must be named either : 'molecule' or 'species' !")

    df = data.copy()

    # Get unique values and convert them to SMILES
    unique_molecules = df['molecule'].unique()
    unique_smiles = inchikey_to_smiles(unique_molecules, n_cpus=n_cpus)

    # Create a dictionary mapping molecules to SMILES
    molecule_to_smiles = dict(zip(unique_molecules, unique_smiles))

    # Map the dictionary onto the 'molecule' column in the dataframe
    df['molecule'] = df['molecule'].map(molecule_to_smiles)
    
    return df

def get_missing_features(data: pd.DataFrame, graph: nx.DiGraph, n_cpus=4):
    mols = []
    for m in data['molecule'].unique():
        if m not in graph:
            mols.append(m)
    
    sp = []
    for s in data['species'].unique():
        if s not in graph:
            sp.append(s)
    
    if len(mols) != 0:
        mols_classyfire = smiles_to_classyfire(mols, n_cpus=n_cpus)
        mols_rdkit = smiles_to_fingerprint(mols)
        mols_rdkit.columns = mols_rdkit.columns.astype(str)
    if len(sp) != 0:
        sp_taxo = get_taxonomy(sp, n_cpus=n_cpus)
    
    #return conditions
    if len(mols) != 0 and len(sp) != 0:
        return mols_classyfire, mols_rdkit, sp_taxo
    elif len(mols) != 0 and len(sp) == 0:
        return mols_classyfire, mols_rdkit, None
    elif len(mols) == 0 and len(sp) != 0:
        return None, None, sp_taxo
    else:
        return None, None, None

def predict(graph : nx.DiGraph,
            model_m_to_s,
            model_s_to_m,
            data: pd.DataFrame,
            molecule_classyfire: pd.DataFrame,
            molecule_fingerprint: pd.DataFrame,
            species_taxo: pd.DataFrame,
            n_cpus=4) -> pd.DataFrame:
    
    #convert molecules that are inchikeys to smiles
    print("Converting Inchikeys to SMILES...")
    data_convert = convert_to_smiles(data=data, n_cpus=n_cpus)
    
    #get missing features
    print("Getting missing features...")
    missing_mols_classyfire, missing_mols_fingerprint, missing_sp_taxo = get_missing_features(data_convert,graph, n_cpus=n_cpus)
    
    #Convert to species binary encoder
    print("Converting species taxonomy as numeric...")
    if missing_sp_taxo is not None:
        sp_feat = pd.concat([species_taxo, missing_sp_taxo])
        species_features = binary_encode_df(sp_feat)
        del sp_feat
    else:
        sp_feat = species_taxo
        species_features = binary_encode_df(sp_feat)
        del sp_feat

    if len(species_features.columns) != 69:
        raise NotImplementedError(f"The model has been trained on 69 features for species, with your species there are now {len(species_features.columns)} features. This is not supported. Please give less species as input.")
    
    #Convert molecules as numeric
    print("Converting molecules as numeric...")
    if missing_mols_classyfire is not None:
        mols_feat = pd.concat([molecule_classyfire, missing_mols_classyfire])
        mols_feat = binary_encode_df(mols_feat)
        mols_feat_fingerprint = pd.concat([molecule_fingerprint, missing_mols_fingerprint])
    else:
        mols_feat = molecule_classyfire
        mols_feat = binary_encode_df(mols_feat)
        mols_feat_fingerprint = molecule_fingerprint
        
    molecule_features = mols_feat.merge(mols_feat_fingerprint,
                                        left_index=True,
                                        right_index=True)
    
    del mols_feat, mols_feat_fingerprint
    if len(molecule_features.columns) != 155:
        raise NotImplementedError(f"The model has been trained on 155 features for molecules, with your molecules there are now {len(molecule_features.columns)} features. This is not supported. Please give less molecules as input.")
    
    
    #first check which model should be used for each row
    print("Checking which model should be used for each row...")
    data_out, m_to_s, s_to_m, both_unknown, both_known = check_which_model_to_use(graph, data_convert)
    
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
        flow_m = create_flow(graph_stellar, data_convert, m_to_s, 'molecule')
        
        print("Predicting mol to species...")
        out_m = _predict(model_m_to_s, flow_m)
        del flow_m
        
    if s_to_m.size != 0:
        print("Creating species to mol flow...")
        flow_s = create_flow(graph_stellar, data_convert, s_to_m, 'species')
        
        print("Predicting species to mol...")
        out_s = _predict(model_s_to_m, flow_s)
        del flow_s

    if both_unknown.size != 0:
        print("Creating 'forward', 'backward' flow for UNKNOWN molecule AND species...")
        flow_m_both_unknown = create_flow(graph_stellar, data_convert, both_unknown, 'molecule')
        flow_s_both_unknown = create_flow(graph_stellar, data_convert, both_unknown, 'species')
        out_both_unknown = _predict_using_both_models(model_m_to_s, model_s_to_m, flow_m_both_unknown, flow_s_both_unknown)
        del flow_m_both_unknown
        del flow_s_both_unknown
        
    if both_known.size != 0:
        print("Creating 'forward', 'backward' flow for KNOWN molecule AND species...")
        flow_m_both_known = create_flow(graph_stellar, data_convert, both_known, 'molecule')
        flow_s_both_known = create_flow(graph_stellar, data_convert, both_known, 'species')
        out_both_known = _predict_using_both_models(model_m_to_s, model_s_to_m, flow_m_both_known, flow_s_both_known)
        del flow_m_both_known
        del flow_s_both_known
    
    if data_convert.columns[0] == 'molecule':
        out_df = pd.DataFrame(np.vstack((m_to_s, s_to_m, both_unknown, both_known)),
                              columns=['molecule', 'species'])
    else:
        out_df = pd.DataFrame(np.vstack((m_to_s, s_to_m, both_unknown, both_known)),
                              columns=['species', 'molecule'])
        
    out_df['prob'] = np.concatenate((out_m, out_s, out_both_unknown, out_both_known))
        
    
    return pd.merge(data_out, out_df, on=['species', 'molecule'])