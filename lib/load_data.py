import numpy as np 
import random
import torch 
from datasets import load_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_data(tokenizer, nsamples=50, seqlen=2048, device=None):
    valdata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")

    num_seq = len(valdata["train"])
    seq_indices = np.random.choice(num_seq, 500, replace=False).tolist()
    seq_list = []
    for seq_ind in seq_indices:
        seq_list.append(valdata["train"][seq_ind]['text'])

    testenc = tokenizer("\n\n".join(seq_list), return_tensors='pt', add_special_tokens=False).input_ids

    testseq_list = []
    for i in range(nsamples):
        test_seq = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
        testseq_list.append(test_seq.reshape(1, seqlen))

    return testseq_list


def get_wikitext(tokenizer, seqlen=2048, device=None):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt', add_special_tokens=False)
    testenc = testenc.input_ids

    testseq_list = []
    nsamples = testenc.numel() // seqlen

    for i in range(nsamples):
        testenc_cur = testenc[:,(i * seqlen):((i+1) * seqlen)].to(device)
        testseq_list.append(testenc_cur.reshape(1, seqlen))
    return testseq_list

def get_pg19(tokenizer, seqlen=2048, device=None):
    valdata = load_dataset(
        'emozilla/pg19', split='validation'
    )

    testseq_list = []
    valenc = tokenizer(' '.join(valdata[:5]['text']), return_tensors='pt').input_ids 
    for i in range(100):
        testseq_list.append(valenc[:, (i * seqlen):((i+1) * seqlen)].to(device))
    return testseq_list

def get_c4(tokenizer, seqlen=2048, device=None):
    valdata = load_dataset("NeelNanda/c4-10k")

    testseq_list = []
    valenc = tokenizer(' '.join(valdata["train"][:5000]['text']), return_tensors='pt').input_ids
    for i in range(100):
        testseq_list.append(valenc[:, (i * seqlen):((i+1) * seqlen)].to(device))
    return testseq_list

def get_test_data(dataset_name, tokenizer=None, seed=0, seqlen=2048, device=None):
    random.seed(seed)
    set_seed(seed)
    if dataset_name == "wikitext":
        return get_wikitext(tokenizer, seqlen=seqlen, device=device)
    elif dataset_name == "c4":
        return get_c4(tokenizer, seqlen=seqlen, device=device)
    elif dataset_name == "pg19":
        return get_pg19(tokenizer, seqlen=seqlen, device=device)
