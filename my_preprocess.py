import numpy as np
import pandas as pd
import pickle

from dgl.data.chem.utils import mol_to_complete_graph,\
                                CanonicalAtomFeaturizer
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, rdmolops
from rdkit import RDConfig

from glob import glob

from xyz2mol import read_xyz_file, xyz2mol
import constants as C


xyz_filepath_list = list(glob(C.RAW_DATA_PATH + 'structures/*.xyz'))
xyz_filepath_list.sort()


## Functions to create the RDKit mol objects
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)

    return mol, np.array(xyz_coordinates), dMat


def bond_featurizer(mol, self_loop=True):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC, None)
            ])
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = torch.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = torch.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1 , 1)

    return bond_feats_dict


class KaggleMolDataset(object):
    def __init__(self, 
                 file_list = xyz_filepath_list,
                 label_filepath = C.RAW_DATA_PATH,
                 store_path = C.PROC_DATA_PATH ,
                 mode='train', 
                 from_raw=True,
                 mol_to_graph = mol_to_bigraph,
                 atom_featurizer=CanonicalAtomFeaturizer,
                 bond_featurizer=bond_featurizer):


        assert mode in ['train', 'test'], \
           'Expect mode to be train or test, got {}.'.format(mode)

        self.mode = mode
        
        self.from_raw = from_raw
        """
        if not from_raw:
            file_name = "%s_processed" % (mode)
        else:
            file_name = "structures"
        self.file_dir = pathlib.Path(file_dir, file_name)
        """
        self.file_list = file_list
        self.store_path = store_path
        self.label_filepath = label_filepath
        self._load(mol_to_graph, atom_featurizer, bond_featurizer)

    def _load(self, mol_to_graph, atom_featurizer, bond_featurizer):
        if not self.from_raw:
            pass
        #    with open(osp.join(self.file_dir, "%s_graphs.pkl" % self.mode), "rb") as f:
        #        self.graphs = pickle.load(f)
        #    with open(osp.join(self.file_dir, "%s_labels.pkl" % self.mode), "rb") as f:
        #        self.labels = pickle.load(f)
        else:
            print('Start preprocessing dataset...')
            labels  = pd.read_csv(self.label_filepath +self.mode + '.csv')
            cnt = 0
            dataset_size = len(labels['molecule_name'].unique())
            mol_names = labels['molecule_name'].unique()
            self.graphs, self.labels = [],[]
            
            for i in range(len(self.file_list)):
                mol_name = self.file_list[i].split('/')[-1][:-4] 
                if mol_name in mol_names:
                    cnt += 1
                    print('Processing molecule {:d}/{:d}'.format(cnt, dataset_size))
                    mol, xyz, dist_matrix = mol_from_xyz(self.file_list[i])
                
                    graph = mol_to_graph(mol, atom_featurizer=atom_featurizer,
                                        bond_featurizer=atom_featurizer)  
                    graph.gdata = {}              
                    smiles = Chem.MolToSmiles(mol)
                    graph.gdata['smile'] = smiles    
                    g.gdata['mol_name'] = mol_name 

                    g.ndata['h'] = torch.stack([g.ndata['h'], xyz], axis = 1)
                    self.graphs.append(graph)
                    label = labels[labels['molecule_name'] ==mol_name ].drop([
                                                                        'molecule_name', 
                                                                        'type',
                                                                        'id'
                                                                      ],
                                                                         axis = 1
                                                                    )
                    self.labels.append(label)

            with open(osp.join(self.store_path, "%s_grapgs.pkl" % self.mode), "wb") as f:
                pickle.dump(self.graphs, f)
            with open(osp.join(self.store_path, "%s_labels.pkl" % self.mode), "wb") as f:
               pickle.dump(self.labels, f)

        print(len(self.graphs), "loaded!")

    def __getitem__(self, item):
        """Get datapoint with index
        Parameters
        ----------
        item : int
            Datapoint index
        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        """
        g, l = self.graphs[item], self.labels[item]
        return g.smile, g, l

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)
