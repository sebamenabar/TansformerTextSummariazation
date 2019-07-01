import os
import sys
import yaml
import time
import random
from copy import deepcopy
from attrdict import AttrDict
from tqdm.autonotebook import tqdm

import rouge
import pickle
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import load_processed_dataframe, get_idxs, CorpusDataset, collate_wrapper, load_vocab
from model import WTransformer
from embeddings import init_from_glove
from utils import AvgMeter, HistoryMeter


def generated_to_string(ys, id2token, padding_idx=0, sos_idx=1, eos_idx=2):
    return [' '.join(id2token[row[(row != padding_idx) & (row != sos_idx) & (row != eos_idx)]]) for row in ys.T]


def greedy_decode(model, src, src_mask, tgt, tgt_mask, vocab_size):
    bsz = src.size(1)
    max_len = tgt.size(0)

    ys = torch.ones(1, bsz, dtype=tgt.dtype, device=tgt.device)

    memory = model.encode(src, src_mask)
    for i in range(max_len):
        output = model.decode(
            tgt=ys, memory=memory, memory_mask=src_mask, tgt_attn_mask=tgt_mask[:i+1, :i+1])
        output = model.vocab_ff(output)
        ys = torch.cat([ys, output.argmax(dim=-1)[-1, :].unsqueeze(0)], dim=0)

    return ys[1:, :].cpu(), output


def train_step(
    model,
    loader,
    optimizer,
    criterion,
    evaluator,
    vocab_size,
    epoch,
    teacher_forcing_ratio,
):
    avg_meter = AvgMeter()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} train')
    now = time.time()
    model.train()
    for i, b in pbar:
        b.src = b.src.to(device, non_blocking=non_blocking)
        b.tgt = b.tgt.to(device, non_blocking=non_blocking)
        b.src_mask = b.src_mask.to(device, non_blocking=non_blocking)
        b.tgt_mask = b.tgt_mask.to(device, non_blocking=non_blocking)
        b.gold = b.gold.to(device, non_blocking=non_blocking)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            output = model(b.src, b.tgt, b.src_mask, b.tgt_mask)
            ys = output.cpu().argmax(dim=-1)
        else:
            ys, output = greedy_decode(
                model, b.src, b.src_mask, b.tgt, b.tgt_mask, vocab_size)

        scores = evaluator.get_scores(
            generated_to_string(ys.numpy(), id2token), b.headline)

        loss = criterion(output.view(-1, vocab_size), b.gold.view(-1))
        if torch.isnan(loss):
            stop = True
            return

        loss.backward()
        optimizer.step()
        model.zero_grad()

        avg_meter.update('loss', loss.item())
        for key, values in scores.items():
            avg_meter.update(key, values)
        avg_meter.inc_count()

        avgs = avg_meter.get_avgs()
        avg_loss = avgs['loss']
        rlf = scores['rouge-l']['f']
        r1f = scores['rouge-1']['f']
        r2f = scores['rouge-2']['f']
        avg_rlf = avgs['rouge-l']['f']
        avg_r1f = avgs['rouge-1']['f']
        avg_r2f = avgs['rouge-2']['f']

        pbar.set_postfix(
            loss=f'{loss.item():.3f} ({avg_loss:.3f})',
            # rouge_l=f'{rlf:.3f} ({avg_rlf:.3f})',
            rouge_1=f'{r1f:.3f} ({avg_r1f:.3f})',
            # rouge_2=f'{r2f:.3f} ({avg_r2f:.3f})',
        )

        pbar.update()

        del output, loss, ys, b

        if i % 100 == 0:
            torch.cuda.empty_cache()

    return time.time() - now, avg_meter.get_avgs()


def val_step(
    model,
    loader,
    optimizer,
    criterion,
    evaluator,
    vocab_size,
    epoch,
    teacher_forcing_ratio,
):
    avg_meter = AvgMeter()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} val')
    now = time.time()
    model.eval()
    with torch.no_grad():
        for i, b in pbar:
            b.src = b.src.to(device, non_blocking=non_blocking)
            b.tgt = b.tgt.to(device, non_blocking=non_blocking)
            b.src_mask = b.src_mask.to(device, non_blocking=non_blocking)
            b.tgt_mask = b.tgt_mask.to(device, non_blocking=non_blocking)
            b.gold = b.gold.to(device, non_blocking=non_blocking)

            ys, output = greedy_decode(
                model, b.src, b.src_mask, b.tgt, b.tgt_mask, vocab_size)

            scores = evaluator.get_scores(
                generated_to_string(ys.numpy(), id2token), b.headline)

            loss = criterion(output.view(-1, vocab_size), b.gold.view(-1))
            if torch.isnan(loss):
                stop = True
                return

            avg_meter.update('loss', loss.item())
            for key, values in scores.items():
                avg_meter.update(key, values)
            avg_meter.inc_count()

            avgs = avg_meter.get_avgs()
            avg_loss = avgs['loss']
            rlf = scores['rouge-l']['f']
            r1f = scores['rouge-1']['f']
            r2f = scores['rouge-2']['f']
            avg_rlf = avgs['rouge-l']['f']
            avg_r1f = avgs['rouge-1']['f']
            avg_r2f = avgs['rouge-2']['f']

            pbar.set_postfix(
                loss=f'{loss.item():.3f} ({avg_loss:.3f})',
                # rouge_l=f'{rlf:.3f} ({avg_rlf:.3f})',
                rouge_1=f'{r1f:.3f} ({avg_r1f:.3f})',
                # rouge_2=f'{r2f:.3f} ({avg_r2f:.3f})',
            )

            pbar.update()

            del output, loss, ys, b

            if i % 100 == 0:
                torch.cuda.empty_cache()

    return time.time() - now, avg_meter.get_avgs()


if __name__ == '__main__':

    with open(sys.argv[1], 'r') as f:
        cfg = AttrDict(yaml.load(f))

    print('Loading data')
    data = load_processed_dataframe(
        fp=os.path.join(cfg.data_dir, 'processed_df.msg'),
        qtl_src_len=getattr(cfg, 'qtl_src_len', None),
        max_src_len=getattr(cfg, 'max_src_len', None),
    )

    train_idxs, val_idxs, test_idxs = get_idxs(
        os.path.join(cfg.data_dir, 'WikiHow/'), data)

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

    if hasattr(cfg, 'model_dir'):
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
    if getattr(cfg, 'use_sample', False):
        with open(os.path.join(cfg.data_dir, 'sample_idxs.pkl'), 'rb') as f:
            train_subset_idxs = pickle.load(f)
        train_ds = Subset(_ds, train_subset_idxs[:100000])
        val_ds = Subset(_ds, train_subset_idxs[100000:])
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=cfg.num_workers,
            batch_size=cfg.bsz,
            collate_fn=collate_wrapper,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            num_workers=cfg.num_workers,
            batch_size=cfg.bsz,
            collate_fn=collate_wrapper,
            pin_memory=use_cuda,
        )
    else:
        train_ds = Subset(_ds, train_idxs)
        val_ds = Subset(_ds, val_idxs)
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=cfg.num_workers,
            batch_size=cfg.bsz,
            collate_fn=collate_wrapper,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            num_workers=cfg.num_workers,
            batch_size=cfg.bsz,
            collate_fn=collate_wrapper,
            pin_memory=use_cuda,
        )

    non_blocking = True
    lr = cfg.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vocab_size = len(token2id)

    epochs = cfg.epochs
    model = model.to(device)

    teacher_forcing_ratio = 0.9
    teacher_forcing_decay = 0.9

    train_history = HistoryMeter()
    val_history = HistoryMeter()

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, apply_avg=True, apply_best=False,
                        alpha=0.5, stemming=True)

    for epoch in tqdm(range(1, epochs + 1)):
        
        train_time, train_avgs = train_step(
            model,
            train_loader,
            optimizer,
            criterion,
            evaluator,
            vocab_size,
            epoch,
            teacher_forcing_ratio,
        )

        print('Epoch train time:', train_time)
        print(train_avgs)

        train_history.update('time', train_time)
        for metric, values in train_avgs.items():
            train_history.update(metric, values)

        val_time, val_avgs = val_step(
            model,
            train_loader,
            optimizer,
            criterion,
            evaluator,
            vocab_size,
            epoch,
            teacher_forcing_ratio,
        )

        val_history.update('time', val_time)
        for metric, values in val_avgs.items():
            val_history.update(metric, values)

        print('Epoch val time:', val_time)
        print(val_avgs)

        teacher_forcing_ratio *= teacher_forcing_decay
        teacher_forcing_ratio = max(teacher_forcing_ratio, 0.5)

        torch.save({
            'model': model.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
        }, os.path.join(model_dir, f'epoch{epoch}.pth'))

