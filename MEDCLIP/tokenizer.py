import json
import os
import torch
from typing import List, Union
from transformers import BertTokenizerFast

# Path to vocab file and special tokens map
VOCAB_PATH = "/kaggle/input/biomedclip/pytorch/default/1/vocab.txt"  # your vocab file path
SPECIAL_TOKENS_MAP_PATH = "/kaggle/input/biomedclip/pytorch/default/1/special_tokens_map.json"

# Load special tokens map
with open(SPECIAL_TOKENS_MAP_PATH, "r") as f:
    special_tokens_map = json.load(f)

# Initialize a BERT tokenizer using the vocab
tokenizer = BertTokenizerFast(
    vocab_file=VOCAB_PATH,
    **special_tokens_map,
    do_lower_case=True,
    model_max_length=512
)

def tokenize(texts: Union[str, List[str]], context_length: int = 256, truncate: bool = True) -> torch.LongTensor:
    """
    Tokenize input text(s) for BiomedCLIP using BERT tokenizer.
    
    Parameters
    ----------
    texts : str or List[str]
        Input string(s) to tokenize
    context_length : int
        Maximum sequence length (BiomedCLIP uses 256 by default)
    truncate : bool
        Whether to truncate sequences longer than context_length
    
    Returns
    -------
    tokens : torch.LongTensor
        Tensor of token ids of shape [num_texts, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    
    encoding = tokenizer(
        texts,
        padding='max_length',
        truncation=truncate,
        max_length=context_length,
        return_tensors='pt'
    )
    return encoding['input_ids']
    

'''# Example usage
sample_texts = ["Liver tumor segmentation.", "CT scan analysis."]
token_ids = tokenize(sample_texts)
print(token_ids.shape)  # should be [2, 256]
print(token_ids[0])'''
