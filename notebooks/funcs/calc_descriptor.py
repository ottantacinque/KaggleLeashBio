import numpy as np
import pandas as pd
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors 
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Fingerprints import FingerprintMols
import gc


def calc_rdkit_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    RDkit = [descriptor_calculation.CalcDescriptors(mol_temp) for mol_temp in mols_list]
    df_RDkit = pd.DataFrame(RDkit, columns = descriptor_names,index=idx_list)

    return df_RDkit


def calc_rdkfp_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    RDkfp = [np.array(FingerprintMols.FingerprintMol(mol), int) for mol in mols_list]
    
    df_RDkfp = pd.DataFrame(RDkfp, index=idx_list)

    return df_RDkfp


def calc_ecfp4_descriptors(bb_dicts:dict, radius:int=2, nBits:int=1024):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    ECFP4 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols_list]
    df_ECFP4 = pd.DataFrame(np.array(ECFP4, int),index=idx_list)
    
    return df_ECFP4


def calc_fcfp4_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    FCFP4 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024, useFeatures=True) for mol in mols_list]
    df_FCFP4 = pd.DataFrame(np.array(FCFP4, int),index=idx_list)
    
    return df_FCFP4


def calc_avalonfp_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    avalonfp = [GetAvalonFP(mol) for mol in mols_list]
    df_avalongp= pd.DataFrame(np.array(avalonfp, int),index=idx_list)
    
    return df_avalongp

def calc_maccs_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mols_list]
    df_maccs = pd.DataFrame(np.array(maccs_fps, int),index=idx_list)
    
    return df_maccs



def calc_mordred_descriptors(bb_dicts: dict, ignore_3D: bool = False, batch_size: int = 100):
    idx_list = list(bb_dicts.keys())
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    
    # 分子を小さなバッチに分割
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    desc = Calculator(descriptors, ignore_3D=ignore_3D)
    
    # 空のデータフレームを初期化
    df_mord_total = None
    
    for idx_batch, smiles_batch in zip(batch(idx_list, batch_size), batch(smiles_list, batch_size)):
        mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch]
        mols_list = [Chem.AddHs(mol) for mol in mols_list]
        
        mols_list_opt = []
        for mol in mols_list:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            mols_list_opt.append(mol)
        
        df_mord_batch = desc.pandas(mols_list_opt, quiet=False)
        df_mord_batch.index = idx_batch
        
        if df_mord_total is None:
            df_mord_total = df_mord_batch
        else:
            df_mord_total = pd.concat([df_mord_total, df_mord_batch])
        
        # メモリを解放
        del mols_list, mols_list_opt, df_mord_batch
        gc.collect()

    return df_mord_total




def calc_path_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    path_to_bb2 = [getPathCount(mol, 0) for mol in mols_list]
    path_to_bb3 = [getPathCount(mol, 1) for mol in mols_list]

    df_path2 = pd.DataFrame(path_to_bb2, columns=["num_atoms_to_bb2", "num_sp3_atoms_bb2", "num_sp3_carbons_bb2"])
    df_path2['ratio_sp3_atoms_bb2'] = df_path2['num_sp3_atoms_bb2'] / df_path2['num_atoms_to_bb2']
    df_path2['ratio_sp3_carbons_bb2'] = df_path2['num_sp3_carbons_bb2'] / df_path2['num_atoms_to_bb2']

    df_path3 = pd.DataFrame(path_to_bb3, columns=["num_atoms_to_bb3", "num_sp3_atoms_bb3", "num_sp3_carbons_bb3"])
    df_path3['ratio_sp3_atoms_bb3'] = df_path3['num_sp3_atoms_bb3'] / df_path3['num_atoms_to_bb3']
    df_path3['ratio_sp3_carbons_bb3'] = df_path3['num_sp3_carbons_bb3'] / df_path3['num_atoms_to_bb3']

    df_path = pd.concat([df_path2, df_path3], axis=1)

    return df_path


def getPathCount(mol, idx):
    """
    指定した原子間の最短経路を取得する
    """
    # # 部分構造の定義
    substructure1 = Chem.MolFromSmiles("C2c3ccccc3-c3ccccc32")  # 部分構造1
    substructure2 = Chem.MolFromSmiles("C(=O)OCCCCCCC")  # 部分構造2

    # 部分構造の検索
    matches1 = mol.GetSubstructMatches(substructure1)
    matches2 = mol.GetSubstructMatches(substructure2)
    if matches1 and matches2:
        atom_idx1 = matches1[idx][0]
        atom_idx2 = matches2[0][0]

        # 最短経路を取得
        shortest_path = GetShortestPath(mol, atom_idx1, atom_idx2)
        
        # sp3原子とSP3炭素の数をカウント
        sp3_atoms_count = 0
        sp3_carbons_count = 0
        for atom_idx in shortest_path:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                sp3_atoms_count += 1
        
            if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3:
                sp3_carbons_count += 1
        
        # 経路上の原子の数
        num_atoms_in_path = len(shortest_path)
  
    else:
        num_atoms_in_path = None
        sp3_atoms_count = None
        sp3_carbons_count = None

    return num_atoms_in_path, sp3_atoms_count, sp3_carbons_count