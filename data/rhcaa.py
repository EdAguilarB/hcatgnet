import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
import os
from tqdm import tqdm
from molvs import standardize_smiles
import sys
from data.datasets import reaction_graph
import re
from sklearn.model_selection import StratifiedKFold


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class rhcaa_diene(reaction_graph):

    def __init__(
        self,
        filename: str,
        molcols: list,
        root: str = None,
        include_Hs=True,
        num_folds=10,
        random_seed=20232023,
    ) -> None:

        self._include_fold = num_folds

        file_folds = filename[:-4] + "_folds" + filename[-4:]

        if self._include_fold:
            if not os.path.exists(os.path.join(root, "raw", f"{file_folds}")):
                self.split_data(root, filename, num_folds, random_seed)
            filename = filename[:-4] + "_folds" + filename[-4:]

        super().__init__(
            filename=filename, mol_cols=molcols, root=root, include_Hs=include_Hs
        )

        self._name = "rhcaa_diene"
        self.get_atom_feats_length()

    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        for index, reaction in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            node_feats_reaction = None

            temp = reaction["temp"] / 100

            mols = {}

            for reactant in self.mol_cols:

                smiles = standardize_smiles(reaction[reactant])

                mols[re.sub(r"\s+", "", reactant).lower()] = smiles

                # create a molecule object from the smiles string
                mol = Chem.MolFromSmiles(standardize_smiles(reaction[reactant]))

                if self._include_Hs:
                    mol = Chem.rdmolops.AddHs(mol)

                node_feats = self._get_node_feats(
                    mol, reaction["Confg"], reactant, temp
                )

                edge_attr, edge_index = self._get_edge_features(mol)

                if node_feats_reaction is None:
                    node_feats_reaction = node_feats
                    edge_index_reaction = edge_index
                    edge_attr_reaction = edge_attr

                else:
                    node_feats_reaction = torch.cat(
                        [node_feats_reaction, node_feats], axis=0
                    )
                    edge_attr_reaction = torch.cat(
                        [edge_attr_reaction, edge_attr], axis=0
                    )
                    edge_index += max(edge_index_reaction[0]) + 1
                    edge_index_reaction = torch.cat(
                        [edge_index_reaction, edge_index], axis=1
                    )

            y = torch.tensor(reaction["ddG"]).reshape(1)
            top = torch.tensor(reaction["%top"]).reshape(1)

            if self._include_fold:
                fold = reaction["fold"]
            else:
                fold = None

            data = Data(
                x=node_feats_reaction,
                edge_index=edge_index_reaction,
                edge_attr=edge_attr_reaction,
                y=y,
                top=top,
                mols=mols,
                idx=index,
                fold=fold,
            )

            torch.save(data, os.path.join(self.processed_dir, f"reaction_{index}.pt"))

    def _get_node_feats(self, mol, mol_confg, reactant, temperature):

        all_node_feats = []
        CIPtuples = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False))

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [0, 2, 3, 4])
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += self._one_h_e(
                self._get_atom_chirality(CIPtuples, atom.GetIdx()),
                ["R", "S"],
                "No_Stereo_Center",
            )
            # feature 7: ligand configuration
            if reactant == "Ligand":
                node_feats += self._one_h_e(mol_confg, [2, 1], 0)
            else:
                node_feats += [0, 0]
            # feature 8: reaction temperature
            node_feats += [temperature]

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)

    def _get_edge_features(self, mol):

        all_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():

            # list to save the edge features
            edge_feats = []

            # Feature 1: Bond type (as double)
            edge_feats += self._one_h_e(
                bond.GetBondType(),
                [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                ],
            )

            # feature 2: double bond stereochemistry
            edge_feats += self._one_h_e(
                bond.GetStereo(),
                [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE],
                Chem.rdchem.BondStereo.STEREONONE,
            )

            # Feature 3: Is in ring
            edge_feats.append(bond.IsInRing())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # create adjacency list
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return torch.tensor(all_edge_feats, dtype=torch.float), edge_indices

    def _create_folds(num_folds, df):
        """
        splits a dataset in a given quantity of folds

        Args:
        num_folds = number of folds to create
        df = dataframe to be splited

        Returns:
        dataset with new "folds" and "mini_folds" column with information of fold for each datapoint
        """

        # Calculate the number of data points in each fold
        fold_size = len(df) // num_folds
        remainder = len(df) % num_folds

        # Create a 'fold' column to store fold assignments
        fold_column = []

        # Assign folds
        for fold in range(1, num_folds + 1):
            fold_count = fold_size
            if fold <= remainder:
                fold_count += 1
            fold_column.extend([fold] * fold_count)

        # Assign the 'fold' column to the DataFrame
        df["fold"] = fold_column

        return df

    def get_atom_feats_length(self):
        self.atom_feats_length = {
            "Atomic Identity": len(self._elem_list),
            "Atom Degree": 4,
            "Atom Hybridization": 4,
            "Atom Aromaticity": 1,
            "Atom in Ring": 1,
            "Atom Chirality": 2,
            "Ligand Confg.": 2,
        }

    def split_data(self, root, filename, n_folds, random_seed):

        dataset = pd.read_csv(os.path.join(root, "raw", f"{filename}"))
        dataset["category"] = dataset["%top"].apply(lambda m: 0 if m < 50 else 1)

        folds = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_seed
        )

        test_idx = []

        for _, test in folds.split(np.zeros(len(dataset)), dataset["category"]):
            test_idx.append(test)

        index_dict = {
            index: list_num
            for list_num, index_list in enumerate(test_idx)
            for index in index_list
        }

        dataset["fold"] = dataset.index.map(index_dict)

        filename = filename[:-4] + "_folds" + filename[-4:]

        dataset.to_csv(os.path.join(root, "raw", filename))

        print("{}.csv file was saved in {}".format(filename, os.path.join(root, "raw")))
