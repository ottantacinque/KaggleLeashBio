
from rdkit import Chem
import numpy as np
# from rdkit.Chem import Draw

# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
# from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

def combine_fragments(main_mol,fragment):
    # 構造を置換する関数

    bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]

    # main_mol の操作 =======================================================
    # main_mol のadjacency matrixの作成と原子の情報を取得する
    main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(main_mol)

    for bond in main_mol.GetBonds():
        main_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
        main_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())

    main_atoms = []
    for atom in main_mol.GetAtoms():
        main_atoms.append(atom.GetSymbol())

    # ダミーアトム（*）の位置を調べる
    r_index_in_main_molecule_old = [index for index, atom in enumerate(main_atoms) if atom == '*']

    # matrix,atom を並び替えて新しく分子を生成する。
    for index, r_index in enumerate(r_index_in_main_molecule_old):
        modified_index = r_index - index
        atom = main_atoms.pop(modified_index)
        main_atoms.append(atom)
        tmp = main_adjacency_matrix[:, modified_index:modified_index + 1].copy()
        main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 1)
        main_adjacency_matrix = np.c_[main_adjacency_matrix, tmp]
        tmp = main_adjacency_matrix[modified_index:modified_index + 1, :].copy()
        main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 0)
        main_adjacency_matrix = np.r_[main_adjacency_matrix, tmp]

    #新しく生成した分子でのダミーアトム(*)の位置を取得する
    r_index_in_main_molecule_new = [index for index, atom in enumerate(main_atoms) if atom == '*']

    #*がどの原子と結合しているか調べる。
    r_bonded_atom_index_in_main_molecule = []
    for number in r_index_in_main_molecule_new:
        r_bonded_atom_index_in_main_molecule.append(np.where(main_adjacency_matrix[number, :] != 0)[0][0])

    #結合のタイプを調べる
    r_bond_number_in_main_molecule = main_adjacency_matrix[
        r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]

    # *原子を削除
    main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 0)
    main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 1)

    for i in range(len(r_index_in_main_molecule_new)):
        main_atoms.remove('*')
    main_size = main_adjacency_matrix.shape[0]
    # Frag_1 の操作ここまで =======================================================

    #Frag_2の操作ここまで===========================================================
    r_number_in_molecule = 0

    #ここから、主骨格の処理と同じ
    fragment_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(fragment)

    for bond in fragment.GetBonds():
        fragment_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(
            bond.GetBondType())
        fragment_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(
            bond.GetBondType())
    fragment_atoms = []
    for atom in fragment.GetAtoms():
        fragment_atoms.append(atom.GetSymbol())

    # integrate adjacency matrix
    r_index_in_fragment_molecule = fragment_atoms.index('*')

    r_bonded_atom_index_in_fragment_molecule = \
        np.where(fragment_adjacency_matrix[r_index_in_fragment_molecule, :] != 0)[0][0]

    # 後で * を削除するのでそのための調整（たぶん）
    if r_bonded_atom_index_in_fragment_molecule > r_index_in_fragment_molecule:
        r_bonded_atom_index_in_fragment_molecule -= 1

    fragment_atoms.remove('*')
    fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 0)
    fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 1)
    #Frag_2の操作ここまで===========================================================

    #Frag_1とFrag_2をつなげる=========================================
    generated_molecule_atoms = main_atoms[:]
    generated_adjacency_matrix = main_adjacency_matrix.copy()

    #新たに生成する分子用のマトリックス作成
    main_size = generated_adjacency_matrix.shape[0]
    generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
        [generated_adjacency_matrix.shape[0], fragment_adjacency_matrix.shape[0]], dtype='int32')]
    generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
        [fragment_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]

    #マトリックスに結合のタイプを記述
    generated_adjacency_matrix[r_bonded_atom_index_in_main_molecule[r_number_in_molecule], 
                               r_bonded_atom_index_in_fragment_molecule + main_size] = \
        r_bond_number_in_main_molecule[r_number_in_molecule]

    generated_adjacency_matrix[r_bonded_atom_index_in_fragment_molecule + main_size, 
                               r_bonded_atom_index_in_main_molecule[r_number_in_molecule]] = \
        r_bond_number_in_main_molecule[r_number_in_molecule]

    generated_adjacency_matrix[main_size:, main_size:] = fragment_adjacency_matrix
    #フラグメントのマトリックスを入力(マトリックスの右下)
    #Frag_1とFrag_2をつなげる=========================================

    # integrate atoms
    generated_molecule_atoms += fragment_atoms

    # generate structures 
    generated_molecule = Chem.RWMol()
    atom_index = []

    for atom_number in range(len(generated_molecule_atoms)):
        atom = Chem.Atom(generated_molecule_atoms[atom_number])
        molecular_index = generated_molecule.AddAtom(atom)
        atom_index.append(molecular_index)

    for index_x, row_vector in enumerate(generated_adjacency_matrix):    
        for index_y, bond in enumerate(row_vector):      
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                generated_molecule.AddBond(atom_index[index_x], atom_index[index_y], bond_list[bond])

    generated_molecule = generated_molecule.GetMol()
    # smiles = Chem.MolToSmiles(generated_molecule)
    # generated_mol = Chem.MolFromSmiles(smiles)

    return generated_molecule




def combine_fragment_st(main_mol, fragment):
    # 構造を置換する関数

    bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
                 Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
                 Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
                 Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
                 Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
                 Chem.rdchem.BondType.ZERO]

    # main_mol の操作
    main_atoms = [atom.GetSymbol() for atom in main_mol.GetAtoms()]

    # ダミーアトム（*）の位置を調べる
    r_index_in_main_molecule_old = [index for index, atom in enumerate(main_atoms) if atom == '*']

    # ダミーアトムの位置を新しい分子に置換
    for r_index in r_index_in_main_molecule_old:
        main_mol = Chem.RWMol(main_mol)
        main_mol.RemoveAtom(r_index)
        main_mol = main_mol.GetMol()

    # フラグメントの操作
    fragment_atoms = [atom.GetSymbol() for atom in fragment.GetAtoms()]

    # ダミーアトムの位置を調べる
    r_index_in_fragment_molecule = fragment_atoms.index('*')

    # ダミーアトムを削除
    fragment = Chem.RWMol(fragment)
    fragment.RemoveAtom(r_index_in_fragment_molecule)
    fragment = fragment.GetMol()

    # フラグメントを結合する
    combined_mol = Chem.CombineMols(main_mol, fragment)
    rw_combined_mol = Chem.RWMol(combined_mol)

    # 結合部位を特定して結合
    main_mol_num_atoms = main_mol.GetNumAtoms()
    fragment_num_atoms = fragment.GetNumAtoms()

    main_mol_last_atom_idx = main_mol_num_atoms - 1
    fragment_first_atom_idx = main_mol_num_atoms

    # 結合を追加
    rw_combined_mol.AddBond(main_mol_last_atom_idx, fragment_first_atom_idx, Chem.rdchem.BondType.SINGLE)

    # 立体化学情報を保持しながら最適化
    combined_mol = rw_combined_mol.GetMol()
    AllChem.EmbedMolecule(combined_mol)
    AllChem.UFFOptimizeMolecule(combined_mol)

    return combined_mol
