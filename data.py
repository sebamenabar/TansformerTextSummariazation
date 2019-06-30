import torch
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_vocab(vocab_dir):
    id2token = pd.read_msgpack(osp.join(vocab_dir, 'id2token.msg'))
    token2id = pd.read_msgpack(osp.join(vocab_dir, 'token2id.msg'))
    return token2id, id2token

def load_processed_dataframe(fp, max_src_len=None, qtl_src_len=None):
    df = pd.read_msgpack(fp)
    # Remove whitespaces
    df.title = df.title.apply(lambda t: t.replace(' ', ''))
    df = df.rename(columns={'data': 'encoded'})

    prev_len = len(df)
    if qtl_src_len is not None:
        quantile = df.data_len.quantile(qtl_src_len)
        df = df[df.data_len <= quantile].reset_index(drop=True)
        print(f'Removed {prev_len - len(df)} examples bigger than quantile {qtl_src_len} ({quantile})')
    elif max_src_len:
        df = df[df.data_len <= max_src_len].reset_index(drop=True)
        print(
            f'Removed {prev_len - len(df)} examples with length bigger than {max_src_len}')

    return df


def get_titles(fp):
    with open(fp, 'r') as f:
        titles = f.readlines()
        titles = [t.strip() for t in titles]
    return pd.Series(titles)

def get_idxs(fp, df, include_unused_to_train=True):
    # fp = '/content/drive/My Drive/U/2019-1/DL/proyecto/WikiHow-Dataset/all_{}.txt'
    train_titles = get_titles(osp.join(fp, 'all_train.txt'))
    val_titles = get_titles(osp.join(fp, 'all_val.txt'))
    test_titles = get_titles(osp.join(fp, 'all_test.txt'))

    train_idxs = df.index[df.title.isin(train_titles)]
    val_idxs = df.index[df.title.isin(val_titles)]
    test_idxs = df.index[df.title.isin(test_titles)]

    print('Total train titles:', len(train_titles))
    print('Total val titles:', len(val_titles))
    print('Total test titles:', len(test_titles))


    print('Train titles not found:', len(
        train_titles[~train_titles.isin(df.title)]))
    print('Val titles not found:', len(
        val_titles[~val_titles.isin(df.title)]))
    print('Test titles not found:', len(
        test_titles[~test_titles.isin(df.title)]))

    print('Unasigned titles:', len(df) - len(train_idxs) - len(val_idxs) - len(test_idxs))

    if include_unused_to_train:
        print('Including unasigned titles in train')
        absent_idxs = df.index[
            ~df.title.isin(train_titles) &
            ~df.title.isin(val_titles) &
            ~df.title.isin(test_titles)
        ]
        train_idxs = train_idxs.append(absent_idxs).sort_values()

    print('Total titles:', len(train_idxs) + len(val_idxs) + len(test_idxs))
    print('Total train titles:', len(train_idxs))
    print('Total val titles:', len(val_idxs))
    print('Total test titles:', len(test_idxs))

    return train_idxs, val_idxs, test_idxs

class CorpusDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]


def generate_square_subsequent_mask(sz):
    return torch.nn.Transformer.generate_square_subsequent_mask(None, sz)


class SimpleCustomBatch:
    def __init__(self, batch):
        self.headline = [b.headline for b in batch]
        self.text = [b.text for b in batch]
        self.src = pad_sequence([torch.from_numpy(b.encoded) for b in batch])
        self.tgt = pad_sequence([torch.from_numpy(b.label) for b in batch])
        self.gold = self.tgt[1:, :]
        self.tgt = self.tgt[:-1, :]
        self.src_mask = (self.src == 0).view(len(batch), -1)
        self.tgt_mask = generate_square_subsequent_mask(self.tgt.size(0))

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.src_mask = self.src_mask.pin_memory()
        self.tgt_mask = self.tgt_mask.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

