from rdkit import Chem
from rdkit.Chem.Draw import MolToImage 
import deepchem as dc 
import torch  


def smiles_to_mol(smiles_string):
    """
    Loads a rdkit molecule object from a given smiles string.
    If the smiles string is invalid, it returns None.
    """
    return Chem.MolFromSmiles(smiles_string)

def mol_file_to_mol(mol_file):
    """
    Checks if the given mol file is valid.
    """
    return Chem.MolFromMolFile(mol_file)

def draw_molecule(mol):
    """
    Draws a molecule in SVG format.
    """
    return MolToImage(mol)

def mol_to_tensor_graph(mol):
    """
    Convert molecule to a graph representation that
    can be fed to the model
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    f = featurizer.featurize(Chem.MolToSmiles(mol))
    data = f[0].to_pyg_graph()
    data["batch_index"] = torch.ones_like(data["x"][:, 0])
    return data

 






