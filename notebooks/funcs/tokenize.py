from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel,BPE
import pandas as pd


def set_tokenizer():
    enc = { '[PAD]':0, '[UNK]':1, 'Br':2, 'C':3, 'N':4, 'O':5, 'H':6, 'S':7, 'F':8, 'Cl':9, 'B':10, 'I':11, 
            's':12, 'o':13, 'c':14, 'n':15, 'i':16, # is Atomic
            '.':17, '=':18, '#':19, # bond
            '/':20, # direction
            '-':21, '+':22, # charge
            '[':23, ']':24, # Atomic mass
            '(':25, ')':26, # Branches
            '@@':27, '@':28, # tetrahedron
            '1':29, '2':30, '3':31, '4':32, '5':33, '6':34, '7':35, '8':36, '9':37
        }

    # トークナイザーの初期化
    tokenizer = Tokenizer(BPE(vocab=enc, unk_token="[UNK]", merges=[('@','@')]))

    # PreTrainedTokenizerFastの初期化
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer_fast.add_special_tokens({'pad_token': '[PAD]', 'sep_token':'[Dy]'})
    tokenizer_fast.add_special_tokens({'additional_special_tokens':['Br', 'Cl']})
    
    return tokenizer_fast


def tokenize_smiles(bb_dicts:dict, max_length:int=80):
    
    tokenizer_fast = set_tokenizer()
    
    idx_list = bb_dicts.keys()
    smiles_list = [bb_dicts[idx] for idx in idx_list]

    token = tokenizer_fast.batch_encode_plus(smiles_list, max_length=max_length, padding='max_length')['input_ids']
    df_token = pd.DataFrame(token, index=idx_list).sort_index()
    
    return df_token

