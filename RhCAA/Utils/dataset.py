import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
from rdkit import Chem
from rdkit.Chem import rdmolops
import os
import torch
from tqdm import tqdm
from molvs import standardize_smiles


def one_h_e(x, allowable_set):
        if x not in allowable_set:
            print(x)
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


class ChiralLigands_class(Dataset):

    def __init__(self, root, filename, include_fold = False,  transform=None, pre_transform=None, pre_filter=None):

        self.filename = filename
        self.include_fold = include_fold
        self.elem_list = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']

        super(ChiralLigands_class, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [f'molecules_{i}_class.pt' for i in list(self.data.index)]
        return molecules
        
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            ###ligand
            ligand = Chem.MolFromSmiles(standardize_smiles(mol["Ligand"]))
            #add Hs
            ligand = rdmolops.AddHs(ligand)
            # Get node features
            node_feats_l = self._get_node_feats(ligand, mol['Confg'])
            # Get edge features
            edge_feats_l = self._get_edge_features(ligand)
            # Get adjacency info
            edge_index_l = self._get_adjacency_info(ligand)
            # Get labels info
            label = self._get_labels(mol["%top"])
            cat = self._get_cat(mol["%top"])

            ###substrate
            substrate = Chem.MolFromSmiles(standardize_smiles(mol["substrate"]))
            #add Hs
            substrate = rdmolops.AddHs(substrate)
            # Get node features
            node_feats_s = self._get_node_feats(substrate, mol['Confg'])
            # Get edge features
            edge_feats_s = self._get_edge_features(substrate)
            # Get adjacency info
            edge_index_s = self._get_adjacency_info(substrate)
            edge_index_s += max(edge_index_l[0]) + 1

            ###boron reagent
            boron = Chem.MolFromSmiles(standardize_smiles(mol["boron reagent"]))
            #add Hs
            boron = rdmolops.AddHs(boron)
            # Get node features
            node_feats_b = self._get_node_feats(boron, mol['Confg'])
            # Get edge features
            edge_feats_b = self._get_edge_features(boron)
            # Get adjacency info
            edge_index_b = self._get_adjacency_info(boron)
            edge_index_b += max(edge_index_s[0]) + 1

            if self.include_fold:
                fold = mol['fold']
            else:
                fold= None

            node_feats = torch.cat((node_feats_l, node_feats_s, node_feats_b), axis = 0)
            edge_feats = torch.cat((edge_feats_l, edge_feats_s, edge_feats_b), axis = 0)
            edge_index = torch.cat((edge_index_l, edge_index_s, edge_index_b), axis = 1)

            chiral_ligand = Data(x=node_feats, 
                                edge_index=edge_index,
                                edge_attr=edge_feats,
                                y=label,
                                category = cat,
                                ligand = standardize_smiles(mol["Ligand"]),
                                substrate = standardize_smiles(mol['substrate']),
                                boron = standardize_smiles(mol["boron reagent"]),
                                ligand_num = mol['ligand_num'],
                                ligand_id = mol['ligand'],
                                ligand_fam = mol['fam'],
                                idx = index,
                                fold = fold
                                ) 

            torch.save(chiral_ligand, 
                        os.path.join(self.processed_dir, 
                        f'molecules_{index}_class.pt'))

    def _get_node_feats(self, mol, mol_confg):

        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += one_h_e(atom.GetSymbol(), self.elem_list)
            # Feature 2: Atom degree
            node_feats += one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Hybridization
            node_feats += one_h_e(atom.GetHybridization(), [0,2,3,4])
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += one_h_e(atom.GetChiralTag(),[0,1,2])
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
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)
    
    
    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):

        molecule = torch.load(os.path.join(self.processed_dir, 
                                f'molecules_{idx}_class.pt')) 
        return molecule



class ChiralLigands_regr(Dataset):

    def __init__(self, root, filename, include_fold = False,  transform=None, pre_transform=None, pre_filter=None):

        self.filename = filename
        self.include_fold = include_fold
        self.elem_list = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']

        super(ChiralLigands_regr, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [f'molecules_{i}_regr.pt' for i in list(self.data.index)]
        return molecules
        
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            ###ligand
            ligand = Chem.MolFromSmiles(standardize_smiles(mol["Ligand"]))
            #add Hs
            ligand = rdmolops.AddHs(ligand)
            # Get node features
            node_feats_l = self._get_node_feats(ligand, mol['Confg'])
            # Get edge features
            edge_feats_l = self._get_edge_features(ligand)
            # Get adjacency info
            edge_index_l = self._get_adjacency_info(ligand)
            # Get labels info
            label = self._get_labels(mol["%top"])
            cat = self._get_cat(mol["%top"])

            ###substrate
            substrate = Chem.MolFromSmiles(standardize_smiles(mol["substrate"]))
            #add Hs
            substrate = rdmolops.AddHs(substrate)
            # Get node features
            node_feats_s = self._get_node_feats(substrate, mol['Confg'])
            # Get edge features
            edge_feats_s = self._get_edge_features(substrate)
            # Get adjacency info
            edge_index_s = self._get_adjacency_info(substrate)
            edge_index_s += max(edge_index_l[0]) + 1

            ###boron reagent
            boron = Chem.MolFromSmiles(standardize_smiles(mol["boron reagent"]))
            #add Hs
            boron = rdmolops.AddHs(boron)
            # Get node features
            node_feats_b = self._get_node_feats(boron, mol['Confg'])
            # Get edge features
            edge_feats_b = self._get_edge_features(boron)
            # Get adjacency info
            edge_index_b = self._get_adjacency_info(boron)
            edge_index_b += max(edge_index_s[0]) + 1

            if self.include_fold:
                fold = mol['fold']
            else:
                fold= None

            
            node_feats = torch.cat((node_feats_l, node_feats_s, node_feats_b), axis = 0)
            edge_feats = torch.cat((edge_feats_l, edge_feats_s, edge_feats_b), axis = 0)
            edge_index = torch.cat((edge_index_l, edge_index_s, edge_index_b), axis = 1)

            chiral_ligand = Data(x=node_feats, 
                                edge_index=edge_index,
                                edge_attr=edge_feats,
                                y=label,
                                category = cat,
                                ligand = standardize_smiles(mol["Ligand"]),
                                substrate = standardize_smiles(mol['substrate']),
                                boron = standardize_smiles(mol["boron reagent"]),
                                ligand_num = mol['ligand_num'],
                                ligand_id = mol['ligand'],
                                ligand_fam = mol['fam'],
                                idx = index,
                                fold = fold
                                ) 

            torch.save(chiral_ligand, 
                        os.path.join(self.processed_dir, 
                        f'molecules_{index}_regr.pt'))
            

    

    def _get_node_feats(self, mol, mol_confg):

        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += one_h_e(atom.GetSymbol(), self.elem_list)
            # Feature 2: Atom degree
            node_feats += one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Hybridization
            node_feats += one_h_e(atom.GetHybridization(), [0,2,3,4])
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += one_h_e(atom.GetChiralTag(),[0,1,2])
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
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)
    
    
    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):

        molecule = torch.load(os.path.join(self.processed_dir, 
                                f'molecules_{idx}_regr.pt')) 
        return molecule
        

