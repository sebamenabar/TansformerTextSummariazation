import os
import sys
import yaml
from copy import deepcopy
from attrdict import AttrDict
from torch.utils.data import DataLoader

from data import load_processed_dataframe, get_idxs, CorpusDataset, collate_wrapper, load_vocab
from model import WTransformer
from embeddings import init_from_glove

if __name__ == '__main__':

    with open(sys.argv[0], 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    print('Loading data')
    data = load_processed_dataframe(
        fp=cfg.data_dir,
        qtl_src_len=cfg.qtl_src_len,
        )
    
    print('Loading vocab')
    token2id, id2token = load_vocab(cfg.data_dir)

    if cfg.use_glove:
        print('Loading embedding')
        _emb = init_from_glove(
            fp=cfg.glove_dir,
            token2id=token2id,
            vocab_size=len(token2id),
            emb_dim=cfg.glove_dim,
        )
    
    print('Creating model')
    emb = deepcopy(_emb)
    model = WTransformer(
        src_emb=emb,
        tgt_emb=emb,
        tgt_vocab_size=len(token2id),
        **cfg.model,
    )

    model_dir = f'glove_{cfg.use_glove}_{cfg.glove_dim}_qtl_{cfg.qtl_src_len}_dm_{cfg.model.d_model}_nh_{cfg.model.n_head}_nel_{cfg.model.num_encoder_layers}_ndl_{cfg.model.num_decoder_layers}_dff_{cfg.model.dim_feedforward}/'
    model_dir = os.path.join(cfg.exp_root, model_dir)
    print('Saving experiments on')
    if os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(model_dir)


