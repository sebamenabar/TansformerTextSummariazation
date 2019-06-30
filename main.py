import os
import sys
import yaml
from copy import deepcopy
from attrdict import AttrDict
from tqdm.autonotebook import tqdm

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import load_processed_dataframe, get_idxs, CorpusDataset, collate_wrapper, load_vocab
from model import WTransformer
from embeddings import init_from_glove


if __name__ == '__main__':

    with open(sys.argv[1], 'r') as f:
        cfg = AttrDict(yaml.load(f))

    print('Loading data')
    data = load_processed_dataframe(
        fp=os.path.join(cfg.data_dir, 'processed_df.msg'),
        qtl_src_len=getattr(cfg, 'qtl_src_len', None),
        max_src_len=getattr(cfg, 'max_src_len', None),
        )

    train_idxs, val_idxs, test_idxs = get_idxs(os.path.join(cfg.data_dir, 'WikiHow/'), data)
    
    print('Loading vocab')
    token2id, id2token = load_vocab(cfg.data_dir)

    if cfg.use_glove:
        print('Loading embedding')
        _emb = init_from_glove(
            dir=cfg.glove_dir,
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

    if hastattr(cfg, 'model_dir'):
        model_dir = os.path.join(cfg.exp_root, cfg.model_dir)
    else:
        if hasattr(cfg, 'qtl_src_len'):
            max_len_str = f'_qtl_{cfg.qtl_src_len * 100:.0f}'
        elif hasattr(cfg, 'max_src_len'):
            max_len_str = f'_len_{cfg.max_src_len}'

        model_dir = f'glove_{cfg.use_glove}_{cfg.glove_dim}{max_len_str}_dm_{cfg.model.d_model}_nh_{cfg.model.nhead}_nel_{cfg.model.num_encoder_layers}_ndl_{cfg.model.num_decoder_layers}_dff_{cfg.model.dim_feedforward}/'
        model_dir = os.path.join(cfg.exp_root, model_dir)
        
    print('Saving experiments on')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(model_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    use_cuda = device == 'cuda'
    if device == 'cuda':
        print('USING GPU BACKEND')

    _ds = CorpusDataset(data)
    train_ds = Subset(_ds, train_idxs)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        num_workers=cfg.num_workers,
        batch_size=cfg.bsz,
        collate_fn=collate_wrapper,
        pin_memory=use_cuda,
    )

    non_blocking = True
    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vocab_size = len(token2id)

    epochs = 10
    stop = False
    i = 0
    model = model.to(device)
    model.train()
    loss_history = []
    for epoch in tqdm(range(1, epochs + 1)):
        avg_loss = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, b in pbar:
            b.src = b.src.to(device, non_blocking=non_blocking)
            b.tgt = b.tgt.to(device, non_blocking=non_blocking)
            b.src_mask = b.src_mask.to(device, non_blocking=non_blocking)
            b.tgt_mask = b.tgt_mask.to(device, non_blocking=non_blocking)
            b.gold = b.gold.to(device, non_blocking=non_blocking)

            model.zero_grad()
            output = model(b.src, b.tgt, b.src_mask, b.tgt_mask)

            loss = criterion(output.view(-1, vocab_size), b.gold.view(-1))
            if torch.isnan(loss):
                stop = True
                break

            loss.backward()
            avg_loss += loss.item()

            optimizer.step()
            i += 1
            pbar.set_postfix(loss=f'{loss.item():.4f} ({avg_loss / (i + 1):.4f})')
            pbar.update()
            
            if i % 500 == 0:
                torch.cuda.empty_cache()

        loss_history.append(avg_loss / (i + 1))

        torch.save({
            'model': model.state_dict(),
            'loss_history': loss_history,
        }, os.path.join(model_dir, f'epoch{epoch}.pth'))

        if stop:
            break



