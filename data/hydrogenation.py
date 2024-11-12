import argparse
import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from tqdm import tqdm
from molvs import standardize_smiles
import networkx as nx
from torch_geometric.utils import from_networkx
import sys
from data.datasets import reaction_graph
from sklearn.model_selection import KFold

from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class hydrogenation_reaction_graph(reaction_graph):

    def __init__(self, opt:argparse.Namespace, filename: str, molcols: list, root: str = None) -> None:

        self._include_fold = opt.split_data

        if self._include_fold:
            if os.path.exists(os.path.join(root, 'raw', f'{filename[:-4]}_folds{filename[-4:]}')):
                file_folds = filename[:-4] + '_folds' + filename[-4:]
                self.folds = pd.read_csv(os.path.join(root, 'raw', f'{file_folds}'))['fold']
            else:
                self.folds = self.split_data(root, filename, opt.folds, opt.global_seed, opt)
                
            filename = filename[:-4] + '_folds' + filename[-4:]
        
        else:
            self.folds = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))[opt.splits_col]

        self.targets = torch.tensor(pd.read_csv(os.path.join(root, 'raw', f'{filename}'))[opt.target_col], dtype=torch.float)
        self.id = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))['limsID'].astype(str)

        super().__init__(opt = opt, filename = filename, mol_cols = molcols, root=root)

        self._name = "hydrogenation"

    @property
    def _elem_list(self):
        elements = [
            'H', 
            'B', 
            'C', 
            'N', 
            'O', 
            'F', 
            'Si', 
            'S', 
            'Cl', 
            'Br',
            'P',
            'K',
            'Li',
            'I',
            'Na',
            'Cs',
            'Fe',
            'Ir']
        
        return elements
        
    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        for index, reaction in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            temp = reaction['temp']/100
            preasure = reaction['preasure']/100
            sc = reaction['S/C']/100

            for reactant in self.mol_cols:  

                #create a molecule object from the smiles string
                if pd.isna(reaction[reactant]) or reaction[reactant] in ('null', ''):
                    mol_graph = self._create_empty_graph()
    
                else:
                    try:
                        smiles = standardize_smiles(reaction[reactant])
                    except:
                        smiles = reaction[reactant]
                    mol_graph = self.smiles_to_graph(smiles)

                    # Ensure edge_index is integer type
                    mol_graph.edge_index = mol_graph.edge_index.to(dtype=torch.long)

                node_feats = mol_graph.x
                rows = node_feats.shape[0]

                # Graph level features
                temp_feat = torch.full((rows, 1), temp)
                preasure_feat = torch.full((rows, 1), preasure)
                sc_feat = torch.full((rows, 1), sc)
                node_feats = torch.cat([node_feats, temp_feat, preasure_feat, sc_feat], axis=1)
                mol_graph.x = node_feats
                mol_graph.id = index
            
                torch.save(mol_graph, 
                        os.path.join(self.processed_dir, 
                                        f'{reactant}_{index}.pt'))
    

    def smiles_to_graph(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)  # Optional for visualization
        G = nx.Graph()
        for atom in mol.GetAtoms():
            atom_features = self._get_atom_features(atom)
            G.add_node(atom.GetIdx(), x=atom_features)  # Atom features
        for bond in mol.GetBonds():
            edge_features = self._get_edge_feats(bond)
            G.add_edge(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()), edge_attr=edge_features)  # Bond features
        data = from_networkx(G)
        data.smiles = smiles
        return data

    def _get_atom_features(self, atom):
        node_feats = []

        # Feature 1: Atomic number
        node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
        # Feature 2: Atom degree
        node_feats += self._one_h_e(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        # Feature 3: Formal Charge
        node_feats += self._one_h_e(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3, 4])
        # Feature 4: Chirality
        node_feats += self._one_h_e(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
                                                          Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
                                                          Chem.rdchem.ChiralType.CHI_OTHER],
                                                          Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        # Feature 5: Num Hs
        node_feats += self._one_h_e(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        # Feature 6: Hybridization
        node_feats += self._one_h_e(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                              Chem.rdchem.HybridizationType.SP,
                                                              Chem.rdchem.HybridizationType.SP2,
                                                              Chem.rdchem.HybridizationType.SP3,
                                                              Chem.rdchem.HybridizationType.SP3D,
                                                              Chem.rdchem.HybridizationType.SP3D2],
                                                              Chem.rdchem.HybridizationType.UNSPECIFIED,)
        # Feature 7: Aromaticity
        node_feats += [atom.GetIsAromatic()]
        # Feature 8: In Ring
        node_feats += [atom.IsInRing()]

        return torch.tensor(node_feats, dtype=torch.float)
    
    def _get_edge_feats(self, bond):
        edge_feats = []

        # Feature 1: Bond type
        edge_feats += self._one_h_e(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, 
                                                         Chem.rdchem.BondType.DOUBLE, 
                                                         Chem.rdchem.BondType.TRIPLE, 
                                                         Chem.rdchem.BondType.AROMATIC, 
                                                         Chem.rdchem.BondType.DATIVE])
        # Feature 2: Double bond stereochemistry
        edge_feats += self._one_h_e(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], Chem.rdchem.BondStereo.STEREONONE)
        # Feature 3: Is in ring
        edge_feats.append(bond.IsInRing())

        return torch.tensor(edge_feats, dtype=torch.float)
    
    def _create_empty_graph(self):
        return Data(x=torch.tensor([0 for _ in range(48)]).unsqueeze(0), edge_index=torch.tensor([[0],[0]]), edge_attr=torch.tensor([0 for _ in range(8)]).unsqueeze(0), smiles = '')


    def len(self):
        return len(self.targets)

    def get(self, idx):
        """
        Retrieves the Data objects for molA, molB, molC, and the target for a given index.
        
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary with Data objects for 'molA', 'molB', 'molC', and the 'target' tensor.
        """
        data = {}

        data['y'] = self.targets[idx]
        data['id'] = self.id[idx]

        for mol in self.mol_cols:
            data[mol] = torch.load(os.path.join(self.processed_dir, f'{mol}_{idx}.pt'))

        return data

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        graph_files = [f'{reactant}_{index}.pt' for index in list(self.data.index) for reactant in self.mol_cols]
        return graph_files

    def split_data(self, root, filename, n_folds, random_seed, opt):

        dataset = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))

        folds = KFold(n_splits = n_folds, shuffle = True, random_state=random_seed)

        test_idx = []

        for _, test in folds.split(np.zeros(len(dataset)), dataset[opt.target_col]):
            test_idx.append(test)

        index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

        dataset['fold'] = dataset.index.map(index_dict)

        filename = filename[:-4] + '_folds' + filename[-4:]

        dataset.to_csv(os.path.join(root, 'raw', filename))

        print('{}.csv file was saved in {}'.format(filename, os.path.join(root, 'raw')))

        return dataset['fold']