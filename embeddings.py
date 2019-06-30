import os.path as osp
import csv
import pandas as pd
import torch
from torch import nn
from tqdm.autonotebook import tqdm
import numpy as np


def init_from_glove(dir, token2id, vocab_size, emb_dim):
    _glove = pd.read_csv(osp.join(dir, f'glove.6B.{emb_dim}d.txt'), sep=" ",
                        index_col=0, header=None, quoting=csv.QUOTE_NONE)

    glove = _glove[_glove.index.isin(token2id.index)]
    print('Vocabulary in glove:', f'{len(glove)}/{len(token2id)}')

    emb = nn.Embedding(num_embeddings=len(token2id), embedding_dim=emb_dim)
    emb.weight.requires_grad = False
    emb.weight[token2id[glove.index].values] = torch.from_numpy(glove.values).to(torch.float32)
    emb.weight.requires_grad = True

    # Preinicializar embeddings, lo que no encuentra quedan aleatorios
    # for token, tok_id in tqdm(token2id.iteritems(), total=len(token2id)):
    #    if token in glove.index:
    #        emb.weight[tok_id] = torch.from_numpy(
    #            glove.loc[token].to_numpy(dtype=np.float32))

    return emb
