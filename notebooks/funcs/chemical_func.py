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
    # TODO:立体構造はできれば保持したい
    
    # smiles→mol（for Debug)
    if "[N+](=O)[O-]" in main_smiles:
        main_smiles = main_smiles.replace("[N+](=O)[O-]", "[Dy]")
    
    # 例外処理等
    # keklize errorが出るもの
    if main_smiles=="O=C(Nc1nc2cc(C(=O)O)ccc2[nH]1)OCC1c2ccccc2-c2ccccc21":
        return "CCOC(=O)c3ccc2[nH]c(c1nc(N)nc(N)n1)nc2c3"
    if main_smiles == "O=C(Nc1nc2ncc(CNc3ccc(C(=O)O)cc3)nc2c(=O)[nH]1)OCC1c2ccccc2-c2ccccc21":
        return "CCOC(=O)c4ccc(NCc3cnc2nc(c1nc(N)nc(N)n1)[nH]c(=O)c2n3)cc4"
    if main_smiles == "[N-]=[N+]=NCCC[C@H](NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O":
        return "CCOC(=O)[C@H](CCCN=N#N)Nc1nc(N)nc(N)n1"
    if main_smiles == 'O=C(N[C@H](Cc1c[nH]c2cc(Cl)ccc12)C(=O)O)OCC1c2ccccc2-c2ccccc21':
        return "CCOC(=O)[C@@H](Cc1c[nH]c2cc(Cl)ccc12)Nc3nc(N)nc(N)n3"
    
    # triazineの根元に5員勘を持つ構造
    if main_smiles == "O=C(O)COC[C@H]1CCCN1C(=O)OCC1c2ccccc2-c2ccccc21":
        return "CCOC(=O)COC[C@H]1CCCN1c2nc(N)nc(N)n2"

    main_mol = Chem.MolFromSmiles(main_smiles)
    
    # フルオレンがなく、# t-butyl escter と フェニルブロもやフェニルヨウ素がある場合
    if not main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1c2ccccc2-c2ccccc21")):
        
        t_butyl = Chem.MolFromSmiles("CC(C)(C)O")
        # br_phenyl = Chem.MolFromSmiles("Brc1ccccc1")
        br_phenyl = Chem.MolFromSmiles("Br")
        i_phenyl = Chem.MolFromSmiles("I")
        Cl_phenyl = Chem.MolFromSmiles("Cl")
        # i_phenyl = Chem.MolFromSmiles("Ic1ccccc1")
        
        # TODO:とりあえずt-butylを削除で対応しているが、正確にやるならメチル基をつけるなどした方がいい
        if main_mol.HasSubstructMatch(t_butyl) and main_mol.HasSubstructMatch(br_phenyl):
            main_mol = AllChem.DeleteSubstructs(main_mol, t_butyl)
            main_mol = replace_structure(main_mol, Chem.MolFromSmiles("Br"), Chem.MolFromSmiles("C[*]"))
            
        if main_mol.HasSubstructMatch(t_butyl) and main_mol.HasSubstructMatch(i_phenyl):
            main_mol = AllChem.DeleteSubstructs(main_mol, t_butyl)
            main_mol = replace_structure(main_mol, Chem.MolFromSmiles("I"), Chem.MolFromSmiles("C[*]"))
            
        if main_mol.HasSubstructMatch(t_butyl) and main_mol.HasSubstructMatch(Cl_phenyl):
            main_mol = AllChem.DeleteSubstructs(main_mol, t_butyl)
            main_mol = replace_structure(main_mol, Chem.MolFromSmiles("Cl"), Chem.MolFromSmiles("C[*]"))
            
    # 例外処理（フルオレンがあるが、triazineが形成されない骨格）
    elif main_mol.HasSubstructMatch(Chem.MolFromSmiles("CN(c1ncccc1C(=O)O)C1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)C1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("O=C(NC1CN(c2cc(C(=O)O)ccn2)C1)OCC1c2ccccc2-c2ccccc21")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("O=C(N[C@@H]1CCN(c2ccccn2)C1)OCC1c2ccccc2-c2ccccc21")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1C=C(c2cccnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1C=C(c2ccncc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1C=C(c2cncnc2)CN1C(=O)OCC1c2ccccc2-c2ccccc21")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1cccnc1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("C1CN(C(=O)OCC2c3ccccc3-c3ccccc32)C[C@H]1c1ccncc1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("c1cccnc1N1CCCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("c1cccnc1N1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)CC1")) or \
        main_mol.HasSubstructMatch(Chem.MolFromSmiles("O=C(OCC1c2ccccc2-c2ccccc21)N1CCC(Cc2ccncc2)(C(=O)O)C1")):
   
        main_mol = AllChem.DeleteSubstructs(main_mol, Chem.MolFromSmiles('C(=O)OCC2c3ccccc3-c3ccccc32'))
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles("Br"), Chem.MolFromSmiles("C[*]"))
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles("I"), Chem.MolFromSmiles("C[*]"))
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles("Cl"), Chem.MolFromSmiles("C[*]"))
        
    elif main_mol.HasSubstructMatch(Chem.MolFromSmiles("CN(C(=O)OCC1c2ccccc2-c2ccccc21)[C@@H](Cc1ccc(Cl)cc1)C(=O)O")):
        # 3級アミン→triazine（idx=23のみ)
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles( "N(C)C(=O)OCC1c2ccccc2-c2ccccc21"), Chem.MolFromSmiles("Nc1nc(N)nc(N(C)[*])n1"))

    # フルオレン→triazide
    else:
        if main_mol.HasSubstructMatch(Chem.MolFromSmiles("N1CCN(C(=O)OCC2c3ccccc3-c3ccccc32)C1")):
            # フルオレンの根元がアミンではない場合
            main_mol = replace_structure(main_mol, Chem.MolFromSmiles( "C(=O)OCC1c2ccccc2-c2ccccc21"), Chem.MolFromSmiles("Nc1nc(N)nc([*])n1"))
        else:
            main_mol = replace_structure(main_mol, Chem.MolFromSmiles( "NC(=O)OCC1c2ccccc2-c2ccccc21"), Chem.MolFromSmiles("Nc1nc(N)nc(N[*])n1"))    
        
        # main_mol = replace_structure(mol, Chem.MolFromSmiles( "C(=O)OCC1c2ccccc2-c2ccccc21"), Chem.MolFromSmiles("Nc1nc(N)nc([*])n1"))
    
    # カルボキシルキ→アミド（linkerとの結合部分）
    if main_mol.HasSubstructMatch(Chem.MolFromSmiles("CC(C)(C)OC(=O)")):
        # t-butyl esterがある場合
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles("C(=O)O"), Chem.MolFromSmiles("*C(=O)N(CC)"), idx=-1)
    else:
        main_mol = replace_structure(main_mol, Chem.MolFromSmiles("C(=O)O([H])"), Chem.MolFromSmiles("*C(=O)N(CC)"))
        
    main_smiles = Chem.MolToSmiles(main_mol)
    
    # smiles→mol（for Debug)
    if "[Dy]" in main_smiles:
        main_smiles = main_smiles.replace("[Dy]", "[N+](=O)[O-]")
            
    return main_smiles