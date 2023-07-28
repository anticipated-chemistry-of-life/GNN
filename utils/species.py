import pygbif
import time
from multiprocessing import Pool
from requests import ConnectTimeout
import pandas as pd
from collections.abc import Iterable

def _get_species_data(i: str) -> dict:
    '''This function will query pygbif for the information about the species. 
    
    NOTE : If it doesn't find a match but finds some alternatives, it will take the first possible alternative'''
    try:
        sp = pygbif.species.name_backbone(name=i, verbose=True)
        if 'alternatives' in sp and 'canonicalName' not in sp:
            return sp['alternatives'][0]
        elif 'alternatives' in sp and 'canonicalName' in sp:
            return sp
        else:
            return sp

    except ConnectionError:
        time.sleep(0.5)
        sp = pygbif.species.name_backbone(name=i, verbose=True)
        if 'alternatives' in sp and 'canonicalName' not in sp:
            return sp['alternatives'][0]
        elif 'alternatives' in sp and 'canonicalName' in sp:
            return sp
        else:
            return sp
        
    except ConnectTimeout:
        time.sleep(0.5)
        sp = pygbif.species.name_backbone(name=i, verbose=True)
        if 'alternatives' in sp and 'canonicalName' not in sp:
            return sp['alternatives'][0]
        elif 'alternatives' in sp and 'canonicalName' in sp:
            return sp
        else:
            return sp

def get_taxonomy(species: Iterable, n_cpus=4) -> pd.DataFrame:
    '''This function will query pygbif for the information about the species. 
    
    NOTE : If it doesn't find a match but finds some alternatives, it will take the first possible alternative'''
    with Pool(processes=n_cpus) as pool:
        ls = pool.map(_get_species_data, species)

    df = pd.DataFrame.from_records(ls)
    df['organism_name'] = species
    return df[['kingdom','phylum','class','order','family','genus', 'canonicalName', 'organism_name']].copy()
