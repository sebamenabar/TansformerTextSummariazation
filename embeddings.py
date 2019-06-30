import csv
import pandas as pd
from torch import nn
from tqdm.autonotebook import tqdm


def init_from_glove(fp, token2id, vocab_size, emb_dim):
    glove = pd.read_csv(f'glove.6B.{emb_dim}d.txt', sep=" ",
                        index_col=0, header=None, quoting=csv.QUOTE_NONE)
    glove = glove.iloc[token2id.index.isin(glove.index)]

    emb = nn.Embedding(num_embeddings=len(token2id), embedding_dim=emb_dim)

    # Preinicializar embeddings, lo que no encuentra quedan aleatorios
    for token, tok_id in tqdm(token2id.iteritems(), total=len(token2id)):
        if token in glove.index:
            emb.weight[tok_id] = torch.from_numpy(
                glove.loc[token].to_numpy(dtype=np.float32))

    return emb
