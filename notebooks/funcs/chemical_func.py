import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors 
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Fingerprints import FingerprintMols
import gc


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



def replace_structure(main_mol, pattern_mol, replace_mol, idx=0):
    
    # patternにマッチする構造を削除して*に置き換える
    if main_mol.HasSubstructMatch(pattern_mol):
        main_mol = AllChem.ReplaceSubstructs(main_mol, pattern_mol, Chem.MolFromSmiles('*'))[idx]
        # *をreplace_molで置き換える
        main_mol = combine_fragments(main_mol, replace_mol)
    
    return main_mol


def clean_bb1_structure(main_smiles):
    
    fluorene = Chem.MolFromSmiles("NC(=O)OCC1c2ccccc2-c2ccccc21")
    fluorene_without_n = Chem.MolFromSmiles("C(=O)OCC2c3ccccc3-c3ccccc32")
    triazine = Chem.MolFromSmiles("Nc1nc(N)nc(N)n1")
    t_butyl_O = Chem.MolFromSmiles("OC(C)(C)(C)")
    metyl = Chem.MolFromSmiles("C")
    br = Chem.MolFromSmiles("Br")
    I = Chem.MolFromSmiles("I")
    Cl = Chem.MolFromSmiles("Cl")
    
    del_t_butyl_halo_patterns = [
    "CC(C)(C)OC(=O)N1CC(c2ccccc2)=C[C@H]1C(=O)O",
    "CC(C)(C)OC(=O)N1CCC(C(=O)O)(c2ccccc2)CC1",
    "CC(C)(C)OC(=O)N1CCC(COc2ccccc2C(=O)O)CC1",
    "CC(C)(C)OC(=O)N1CCC(COc2ccc(C(=O)O)cc2)CC1",
    "CC(C)(C)OC(=O)N1CCC(Oc2cc(C(=O)O)ccc2I)CC1",
    "CC(C)(C)OC(=O)N1CCC[C@@H](c2ccccc2)[C@@H]1C(=O)O",
    "CC(C)(C)OC(=O)N1CCC[C@@H](n2cccc2C(=O)O)C1",
    "CC(C)(C)OC(=O)N1CCC[C@H](n2cc(C(=O)O)c3ccccc32)C1",
    "CC(C)(C)OC(=O)N1CC[C@@](Cc2cccs2)(C(=O)O)C1",
    "CC(C)(C)OC(=O)N1CC[C@H](Oc2ccccc2C(=O)O)C1",
    "CC(C)(C)OC(=O)N1CC[C@](Cc2ccccc2)(C(=O)O)C1",
    "CC(C)(C)OC(=O)N1C[C@@H](C(=O)O)[C@H](c2ccccc2)C1",
    "CC(C)(C)OC(=O)N1C[C@@H](Oc2ccccn2)C[C@@H]1C(=O)O",
    "CC(C)(C)OC(=O)N1C[C@@H](n2cncc2)C[C@H]1C(=O)O",
    "CC(C)(C)OC(=O)N1C[C@H](Oc2ccccc2)C[C@@H]1C(=O)O",
    "CC(C)(C)OC(=O)N[C@@H]1CCCN(c2ncccc2C(=O)O)C1",
    "CN(c1cc(C(=O)O)ccn1)C1CCN(C(=O)OC(C)(C)C)C1",
    ]
    del_fluorene_halo_patterns = [
        "CN(c1ncccc1C(=O)O)C1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)C1",
        "O=C(NC1CN(c2cc(C(=O)O)ccn2)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(N[C@@H]1CCN(c2cc(C(=O)O)ccn2)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(N[C@@H]1CCN(c2ncccc2C(=O)O)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2cccnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2ccncc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2cncnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1cccnc1",
        "O=C(O)[C@@H]1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1ccncc1",
        "O=C(O)c1cccnc1N1CCCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1",
        "O=C(O)c1cccnc1N1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1",
        "O=C(O)c1ccnc(N2CCCN(C(=O)OCC3c4ccccc4-c4ccccc43)CC2)c1",
        "O=C(O)c1ccnc(N2CCN(C(=O)OCC3c4ccccc4-c4ccccc43)CC2)c1",
        "O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)C1",
        "O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)CC1",
    ]
    ex_dict = {}
    
    # 例外処理
    if main_smiles in ex_dict.keys():
        return ex_dict[main_smiles]
    
    main_mol = Chem.MolFromSmiles(main_smiles)
    
    # fluoreneがなく、triazineにならないもの
    for pattern in del_t_butyl_halo_patterns:
        pattern_mol = Chem.MolFromSmiles(pattern)
        if main_mol.HasSubstructMatch(pattern_mol):
            main_mol = AllChem.ReplaceSubstructs(main_mol, t_butyl_O, metyl)[0]
            # main_mol = AllChem.DeleteSubstructs(main_mol, t_butyl_O)
            main_mol = AllChem.ReplaceSubstructs(main_mol, br, metyl)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, I, metyl)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, Cl, metyl)[0]
            
    for pattern in del_fluorene_halo_patterns:
        pattern_mol = Chem.MolFromSmiles(pattern)
        if main_mol.HasSubstructMatch(pattern_mol):
            main_mol = AllChem.ReplaceSubstructs(main_mol, fluorene_without_n, metyl)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, br, metyl)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, I, metyl)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, Cl, metyl)[0]
    
    # fluorene→triazine
    main_mol = AllChem.ReplaceSubstructs(main_mol, fluorene, triazine)[0]
 
    main_smiles = Chem.MolToSmiles(main_mol)
            
    return main_smiles


def clean_bb23_structure(main_smiles):

    remove_list_on_smiles = [".Cl", ".Br", ".I", "Cl.", "Br.", "I.", ".O=C(O)C(=O)O", "O=C(O)C(=O)O."]
    replace_list_on_smiles = [
        ("B1OC(C)(C)C(C)(C)O1", "C"),
        ("B2OCCCO2", "C"),
        ("B(O)O", "C"),
                            ]

    # smilesで処理
    for smiles in remove_list_on_smiles:
        main_smiles = main_smiles.replace(smiles, "")
        
        
    main_mol = Chem.MolFromSmiles(main_smiles)
    
    for smiles in replace_list_on_smiles:
        pattern_mol = Chem.MolFromSmiles(smiles[0])
        replace_mol = Chem.MolFromSmiles(smiles[1])
        main_mol = AllChem.ReplaceSubstructs(main_mol, pattern_mol, replace_mol)[0]    
    
    main_smiles = Chem.MolToSmiles(main_mol)
            
    return main_smiles



def clean_and_capping_bb1_structure(main_smiles):
        
    del_t_butyl_halo_patterns = [
        "CC(C)(C)OC(=O)N1CC(c2ccccc2)=C[C@H]1C(=O)O",
        "CC(C)(C)OC(=O)N1CCC(C(=O)O)(c2ccccc2)CC1",
        "CC(C)(C)OC(=O)N1CCC(COc2ccccc2C(=O)O)CC1",
        "CC(C)(C)OC(=O)N1CCC(COc2ccc(C(=O)O)cc2)CC1",
        "CC(C)(C)OC(=O)N1CCC(Oc2cc(C(=O)O)ccc2I)CC1",
        "CC(C)(C)OC(=O)N1CCC[C@@H](c2ccccc2)[C@@H]1C(=O)O",
        "CC(C)(C)OC(=O)N1CCC[C@@H](n2cccc2C(=O)O)C1",
        "CC(C)(C)OC(=O)N1CCC[C@H](n2cc(C(=O)O)c3ccccc32)C1",
        "CC(C)(C)OC(=O)N1CC[C@@](Cc2cccs2)(C(=O)O)C1",
        "CC(C)(C)OC(=O)N1CC[C@H](Oc2ccccc2C(=O)O)C1",
        "CC(C)(C)OC(=O)N1CC[C@](Cc2ccccc2)(C(=O)O)C1",
        "CC(C)(C)OC(=O)N1C[C@@H](C(=O)O)[C@H](c2ccccc2)C1",
        "CC(C)(C)OC(=O)N1C[C@@H](Oc2ccccn2)C[C@@H]1C(=O)O",
        "CC(C)(C)OC(=O)N1C[C@@H](n2cncc2)C[C@H]1C(=O)O",
        "CC(C)(C)OC(=O)N1C[C@H](Oc2ccccc2)C[C@@H]1C(=O)O",
        "CC(C)(C)OC(=O)N[C@@H]1CCCN(c2ncccc2C(=O)O)C1",
        "CN(c1cc(C(=O)O)ccn1)C1CCN(C(=O)OC(C)(C)C)C1",
        ]
    del_fluorene_halo_patterns = [
        "CN(c1ncccc1C(=O)O)C1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)C1",
        "O=C(NC1CN(c2cc(C(=O)O)ccn2)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(N[C@@H]1CCN(c2cc(C(=O)O)ccn2)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(N[C@@H]1CCN(c2ncccc2C(=O)O)C1)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2cccnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2ccncc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1C=C(c2cncnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21",
        "O=C(O)[C@@H]1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1cccnc1",
        "O=C(O)[C@@H]1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1ccncc1",
        "O=C(O)c1cccnc1N1CCCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1",
        "O=C(O)c1cccnc1N1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1",
        "O=C(O)c1ccnc(N2CCCN(C(=O)OCC3c4ccccc4-c4ccccc43)CC2)c1",
        "O=C(O)c1ccnc(N2CCN(C(=O)OCC3c4ccccc4-c4ccccc43)CC2)c1",
        "O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)C1",
        "O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)CC1",
    ]
    tert_butyl_ester_smiles = ["CC(C)(C)OC(=O)CC(NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O",
                            "CC(C)(C)OC(=O)CCC(NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O",
                            "CC(C)(C)OC(=O)N1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)C1C(=O)O",
                            "CC(C)(C)OC(=O)N1C[C@@H](NC(=O)OCC2c3ccccc3-c3ccccc32)[C@H](C(=O)O)C1",
                            "COC(=O)CC(NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O",
                            "COC(=O)c1ccc(C(=O)O)c(NC(=O)OCC2c3ccccc3-c3ccccc32)c1",
                            ]
    
    fluorene = Chem.MolFromSmiles("NC(=O)OCC1c2ccccc2-c2ccccc21")
    fluorene_without_n = Chem.MolFromSmiles("C(=O)OCC2c3ccccc3-c3ccccc32")
    fluorene_scaffold = Chem.MolFromSmiles("CC3c1ccccc1c2ccccc23")
    triazine = Chem.MolFromSmiles("Nc1nc(N)nc(N)n1")
    triazine_and_fluorene = Chem.MolFromSmiles("Nc7nc(NC3c1ccccc1c2ccccc23)nc(NC6c4ccccc4c5ccccc56)n7")
    
    t_butyl_O = Chem.MolFromSmiles("COC(C)(C)(C)")
    carboxyl = Chem.MolFromSmiles("C(=O)O")
    propane_ester = Chem.MolFromSmiles("C(=O)OCCCCCCC")
    metyl = Chem.MolFromSmiles("C")
    br = Chem.MolFromSmiles("Br")
    I = Chem.MolFromSmiles("I")
    Cl = Chem.MolFromSmiles("Cl")
    
    # 例外処理
    if main_smiles in ex_dict.keys():
        return ex_dict[main_smiles]
    
    main_mol = Chem.MolFromSmiles(main_smiles)
    
    # fluoreneがなく、triazineにならないもの
    for pattern in del_t_butyl_halo_patterns:
        pattern_mol = Chem.MolFromSmiles(pattern)
        if main_mol.HasSubstructMatch(pattern_mol):
            main_mol = AllChem.ReplaceSubstructs(main_mol, t_butyl_O, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, br, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, I, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, Cl, fluorene_scaffold)[0]
            # main_mol = AllChem.DeleteSubstructs(main_mol, t_butyl_O)
            
            
            main_mol = AllChem.ReplaceSubstructs(main_mol, carboxyl, propane_ester)[0]
            
            main_smiles = Chem.MolToSmiles(main_mol)    
            return main_smiles
    
    # fluoreneはあるがtriaineにならないもの
    for pattern in del_fluorene_halo_patterns:
        pattern_mol = Chem.MolFromSmiles(pattern)
        if main_mol.HasSubstructMatch(pattern_mol):
            main_mol = AllChem.ReplaceSubstructs(main_mol, fluorene_without_n, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, br, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, I, fluorene_scaffold)[0]
            main_mol = AllChem.ReplaceSubstructs(main_mol, Cl, fluorene_scaffold)[0]
            
            main_mol = AllChem.ReplaceSubstructs(main_mol, carboxyl, propane_ester)[0]
            main_smiles = Chem.MolToSmiles(main_mol)    
            return main_smiles

    
    # fluorene→triazine
    main_mol = AllChem.ReplaceSubstructs(main_mol, fluorene, triazine_and_fluorene)[0]
    
    # carboxylをpropane_ester
    if main_smiles in tert_butyl_ester_smiles:
        main_mol = AllChem.ReplaceSubstructs(main_mol, carboxyl, propane_ester)[1]        
    else:
        main_mol = AllChem.ReplaceSubstructs(main_mol, carboxyl, propane_ester)[0]
 
    main_smiles = Chem.MolToSmiles(main_mol)
            
    return main_smiles