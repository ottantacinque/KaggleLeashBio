import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors 
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


def calc_ecfp4_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    ECFP4 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2,1024) for mol in mols_list]
    df_ECFP4 = pd.DataFrame(np.array(ECFP4, int),index=idx_list)
    return df_ECFP4


def calc_fcfp4_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    FCFP4 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024,useFeatures=True) for mol in mols_list]
    df_FCFP4 = pd.DataFrame(np.array(FCFP4, int),index=idx_list)
    
    return df_FCFP4


def calc_maccs_descriptors(bb_dicts:dict):

    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mols_list]
    df_maccs = pd.DataFrame(np.array(maccs_fps, int),index=idx_list)
    
    return df_maccs


# def calc_mordred_descriptors(bb_dicts:dict, ignore_3D:bool=False):

#     idx_list = bb_dicts.keys()
#     smiles_list = [bb_dicts[idx] for idx in idx_list]
    
#     mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
#     mols_list = [Chem.AddHs(mol) for mol in mols_list]

#     mols_list_opt = []
#     for mol in mols_list:
#         AllChem.EmbedMolecule(mol, AllChem.ETKDG())
#         mols_list_opt.append(mol)

#     desc = Calculator(descriptors, ignore_3D=ignore_3D) #3D記述子
#     df_mord = desc.pandas(mols_list_opt, quiet=False)
#     df_mord.index = idx_list
    
#     del mols_list, mols_list_opt
#     gc.collect()
    
#     return df_mord

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