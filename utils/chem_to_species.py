import requests
import json
from urllib.parse import quote
import pandas as pd
import pubchempy
from utils.molecules import is_inchikey
import networkx as nx

def get_pubchem_data(cid):
    query = {
        "select": "*",
        "collection": "consolidatedcompoundtaxonomy",
        "where": {
            "ands": [
                {"cid": cid}
            ]
        },
        "order": ["cid,asc"],
        "start": 1,
        "limit": 10000
    }
    
    # Use the json library's built-in dumps function to convert the dictionary to a JSON string
    query_string = quote(json.dumps(query), safe='')

    # Insert the JSON string into the URL
    url = f'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?outfmt=json&query={query_string}'

    response = requests.get(url)
    # Check that the request was successful
    if response.status_code == 200:
        x = response.json()
        return pd.DataFrame.from_records(x['SDQOutputSet'][0]['rows'])  # If successful, return the data in JSON format
    else:
        return None  # If the request was not successful, return None

def _inchikey_to_compound(inchikey: str) -> pubchempy.Compound:
    '''Input must be either InchiKey or a SMILES.
    If multiple compounds return from the query (which normally should not be the case), it will take the first one.
    '''
    if not is_inchikey(inchikey):
        return inchikey
    
    #fetch compound in pubchem database
    list_compound = pubchempy.get_compounds(inchikey, 'inchikey')
    
    if len(list_compound) == 0:
        raise ValueError(f'{inchikey} not found in PubChem ! Please specify only Inchikeys that are valid in PubChem.')
    
    #return canonical smiles of the molecule
    return list_compound[0]

def _inchikey_to_cid(inchikey: str)-> int:
    compound = _inchikey_to_compound(inchikey)
    return compound.cid

def add_missing_edges(graph: nx.DiGraph, cid: int):
    return 0
    