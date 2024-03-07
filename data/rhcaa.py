import argparse
import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
import os
from tqdm import tqdm
from molvs import standardize_smiles
import sys
from data.datasets import reaction_graph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class rhcaa_diene(reaction_graph):

    def __init__(self, opt:argparse.Namespace, filename: str, molcols: list, root: str = None, include_fold = True) -> None:
        self._include_fold = include_fold
        super().__init__(opt = opt, filename = filename, mol_cols = molcols, root=root)

        self._name = "rhcaa_diene"
        


    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        for index, reaction in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            node_feats_reaction = None

            for reactant in self.mol_cols:  

                #create a molecule object from the smiles string
                mol = Chem.MolFromSmiles(standardize_smiles(reaction[reactant]))

                mol = Chem.rdmolops.AddHs(mol)

                node_feats = self._get_node_feats(mol, reaction['Confg'])

                edge_attr, edge_index = self._get_edge_features(mol)

                if node_feats_reaction is None:
                    node_feats_reaction = node_feats
                    edge_index_reaction = edge_index
                    edge_attr_reaction = edge_attr

                else:
                    node_feats_reaction = torch.cat([node_feats_reaction, node_feats], axis=0)
                    edge_attr_reaction = torch.cat([edge_attr_reaction, edge_attr], axis=0)
                    edge_index += max(edge_index_reaction[0]) + 1
                    edge_index_reaction = torch.cat([edge_index_reaction, edge_index], axis=1)

            label = torch.tensor(reaction['%top']).reshape(1)

            if self._include_fold:
                fold = reaction['fold']
            else:
                fold = None

            data = Data(x=node_feats_reaction, 
                        edge_index=edge_index_reaction, 
                        edge_attr=edge_attr_reaction, 
                        y=label,
                        ligand = standardize_smiles(reaction['Ligand']),
                        substrate = standardize_smiles(reaction['substrate']),
                        boron = standardize_smiles(reaction['boron reagent']),
                        ligand_num = reaction['ligand_num'],
                        ligand_id = reaction['ligand'],
                        idx = index,
                        fold = fold
                        ) 
            
            torch.save(data, 
                       os.path.join(self.processed_dir, 
                                    f'molecules_{index}_regr.pt'))

            if index % 100 == 0:
                print('Reaction {} processed and saved as reaction_{}.pt'.format(index, index))
            

    

    def _get_node_feats(self, mol, mol_confg):

        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [0,2,3,4])
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += self._one_h_e(atom.GetChiralTag(),[0,1,2])
            #feature 7: mol configuration
            node_feats.append(mol_confg)

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)
    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)
    
    def _get_cat(self, label):
        label = np.asarray(label)
        if label <= 50:
            cat = [0]
        else:
            cat = [1]
        return torch.tensor(cat, dtype=torch.int64)


    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    def _get_edge_features(self, mol):

        all_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():

            #list to save the edge features
            edge_feats = []

            # Feature 1: Bond type (as double)
            edge_feats += self._one_h_e(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])

            #feature 2: double bond stereochemistry
            edge_feats += self._one_h_e(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], Chem.rdchem.BondStereo.STEREONONE)

            # Feature 3: Is in ring
            edge_feats.append(bond.IsInRing())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #create adjacency list
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return torch.tensor(all_edge_feats, dtype=torch.float), edge_indices
    
