import io
import requests
import pandas as pd
import numpy as np
import pubchempy
import re
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool
from collections.abc import Iterable

URL = "http://classyfire.wishartlab.com"


def _structure_query(compound: str, label="pyclassyfire") -> int:
    """Submit a compound information to the ClassyFire service for evaluation
    and receive a id which can be used to used to collect results

    :param compound: The compound structures as line delimited inchikey or
         smiles. Optionally a tab-separated id may be prepended for each
         structure.
    :type compound: str
    :param label: A label for the query
    :type label:
    :return: A query ID number
    :rtype: int

    >>> structure_query('CCC', 'smiles_test')
    >>> structure_query('InChI=1S/C3H4O3/c1-2(4)3(5)6/h1H3,(H,5,6)')

    """
    r = requests.post(
        URL + "/queries.json",
        data='{"label": "%s", '
        '"query_input": "%s", "query_type": "STRUCTURE"}' % (label, compound),
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()["id"]


def _get_result(query_id: int, return_format="csv") -> str:
    """Given a query_id, fetch the classification results
    :param query_id: A numeric query id returned at time of query submission
    :type query_id: str
    :param return_format: desired return format. valid types are json, csv or sdf
    :type return_format: str
    :return: query information
    :rtype: st
    >>> get_results('595535', 'csv')
    >>> get_results('595535', 'json')
    >>> get_results('595535', 'sdf'
    """
    r = requests.get(
        "%s/queries/%s.%s" % (URL, query_id, return_format),
        headers={"Content-Type": "application/%s" % return_format},
    )
    r.raise_for_status()
    return r.text


def _convert_to_csv(result: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(result))


def _filter_df(df: pd.DataFrame, compound: str) -> pd.DataFrame:
    """Function that parses the result from classyfire and looks for  'Kingdom', 'Superclass', 'Class', 'Direct_parent'
    of the compound"""
    # Define the desired classes
    classes = ["Kingdom", "Superclass", "Class", "Direct_parent"]

    # Filter dataframe to only include rows where the 'ClassifiedResults' column starts with one of the desired classes
    df_filtered = df[
        df["ClassifiedResults"].str.split(": ").str[0].isin(classes)
    ].copy()

    # Split 'ClassifiedResults' into two separate columns
    df_filtered[["Classification", "Result"]] = df_filtered[
        "ClassifiedResults"
    ].str.split(": ", expand=True)

    # Set 'Classification' as index and transpose dataframe
    df_filtered = df_filtered.set_index("Classification").transpose()

    # Drop unnecessary columns and rows
    df_filtered = df_filtered.drop(["CompoundID", "ChemOntID", "ClassifiedResults"])

    # Rename columns
    df_filtered = df_filtered.rename(
        columns={
            "Kingdom": "structure_taxonomy_classyfire_01kingdom",
            "Superclass": "structure_taxonomy_classyfire_02superclass",
            "Class": "structure_taxonomy_classyfire_03class",
            "Direct_parent": "structure_taxonomy_classyfire_04directparent",
        }
    )

    df_filtered.index = [compound]
    return df_filtered


def _smiles_to_classyfire(compound: str) -> pd.DataFrame:
    """Function that takes a smiles as input and return the classified compound in the different pathway."""
    # first fetch compound ID
    compound_id = _structure_query(compound)

    # get its calssification as string
    result_string = _get_result(compound_id)

    # convert it into a dataframe
    class_df = _convert_to_csv(result_string)

    # filter it
    filtered_df = _filter_df(class_df, compound=compound)

    return filtered_df


def smiles_to_classyfire(compounds: Iterable, n_cpus=4) -> pd.DataFrame:
    with Pool(processes=n_cpus) as pool:
        ls = pool.map(_smiles_to_classyfire, compounds)
    return pd.concat(ls)


def is_inchikey(string: str) -> bool:
    """
    Checks if a string is a valid InChIKey.

    The InChIKey is a 27-character string, consisting of 14 characters, a hyphen,
    10 characters, a hyphen, and a single character.
    """
    # Regex pattern for InChIKey
    pattern = "^[A-Z]{14}-[A-Z]{10}-[A-Z]$"

    # Use the re.match function to check if the string matches the pattern
    if re.match(pattern, string):
        return True
    else:
        return False


def _inchikey_to_smiles(inchikey: str) -> str:
    """Input must be either InchiKey or a SMILES.
    If multiple compounds return from the query (which normally should not be the case), it will take the first one.
    """
    if not is_inchikey(inchikey):
        return inchikey

    # fetch compound in pubchem database
    list_compound = pubchempy.get_compounds(inchikey, "inchikey")

    if len(list_compound) == 0:
        return f"{inchikey} not found in PubChem !"

    # return canonical smiles of the molecule
    return list_compound[0].canonical_smiles


def inchikey_to_smiles(inchikeys: Iterable[str], n_cpus=4) -> list:
    """Takes a list of inchikeys and returns the SMILES, in parallel"""
    with Pool(processes=n_cpus) as pool:
        ls = pool.map(_inchikey_to_smiles, inchikeys)
    return ls


def _to_bit_vec(mol) -> np.ndarray:
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(mol, fp)
    return fp


# def _smiles_to_fingerprint(smile: str, radius=2, nBits=1024) -> np.ndarray:
#    fps = AllChem.MolFromSmiles(smile)
#    mol = AllChem.GetMorganFingerprintAsBitVect(fps, radius=radius, nBits=nBits)
#    return _to_bit_vec(mol)


def smiles_to_fingerprint(smiles: Iterable, radius=2, nBits=128) -> pd.DataFrame:
    """
    Converts SMILES to molecular fingerprint
    """
    fps = [AllChem.MolFromSmiles(i) for i in smiles]
    mols = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        for m in fps
    ]
    mol_dum = [_to_bit_vec(i) for i in mols]
    mol_dum = pd.DataFrame.from_records(mol_dum).astype("uint8")
    mol_dum.index = [i for i in smiles]

    return mol_dum
