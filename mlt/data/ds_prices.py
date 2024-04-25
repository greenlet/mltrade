from datetime import datetime
from pathlib import Path
from typing import Generator, Union, Optional

import numpy as np
import pandas as pd
import torch


def mask_to_zero(rng: np.random.Generator, a: np.ndarray, min_sz: int, max_sz: int) -> np.ndarray:
    n, m = a.shape
    assert max_sz <= m
    b = a.copy()
    sizes = rng.integers(min_sz, max_sz + 1, size=n)
    for i in range(n):
        inds = rng.choice(m, sizes[i], replace=False)
        b[i, inds] = 0
    return b


def row_to_timestamp(row):
    date_, time_ = int(row.Date), int(row.Time)
    dt = datetime(
        year=date_ // 10000, month=(date_ // 100) % 100, day=date_ % 100,
        hour=time_ // 10000, minute=(time_ // 100) % 100, second=time_ % 100,
    )
    return dt.timestamp()


def discount_from_mask(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    res = np.ones(n)
    ind = 0
    while ind < n:
        if mask[ind]:
            ind1 = ind
            while ind1 < n and mask[ind1]:
                ind1 += 1
            off = ind1 - ind
            if ind == 0:
                assert ind1 < n
                for i in range(off):
                    res[ind1 - 1 - i] = i + 1
            elif ind1 == n:
                for i in range(off):
                    res[ind + i] = i + 1
            else:
                j, up = 0, True
                for i in range(off):
                    if up:
                        j += 1
                        off2 = off // 2
                        if j > off2:
                            j = off2 + off % 2
                            up = False
                    else:
                        j -= 1
                    res[ind + i] = j
            ind = ind1
        else:
            ind += 1
    res = np.sqrt(res)
    return res


class BatchResType:
    inp: np.ndarray
    tgt: np.ndarray
    div: np.ndarray
    inp_t: Optional[torch.Tensor]
    tgt_t: Optional[torch.Tensor]
    div_t: Optional[torch.Tensor]

    def __init__(self, inp: np.ndarray, tgt: np.ndarray, div: np.ndarray, inp_t: Optional[torch.Tensor] = None,
                 tgt_t: Optional[torch.Tensor] = None, div_t: Optional[torch.Tensor] = None):
        self.inp = inp
        self.tgt = tgt
        self.div = div
        self.inp_t = inp_t
        self.tgt_t = tgt_t
        self.div_t = div_t


BatchGenType = Generator[BatchResType, None, None]


class DsPrices:
    fpath: Path
    df: pd.DataFrame
    prices_names: list[str]
    yields_names: list[str]
    train_ratio: float
    n_train: int
    n_val: int
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    prices_train: np.ndarray
    prices_val: np.ndarray
    batch_size: int
    min_inp_size: int
    max_inp_size: int
    min_inp_zeros: int
    max_inp_zeros_val: int
    rng: np.random.Generator

    def __init__(self, fpath: Path, batch_size: int, min_inp_size: int, max_inp_size: int,
                 min_inp_zeros: int, max_inp_zeros_rate: float, train_ratio: float = 0.9):
        self.fpath = fpath
        self.df = pd.read_csv(self.fpath, sep=',', header=0, dtype=np.float64)
        self.prices_names = [cn for cn in self.df.columns if cn not in ('Date', 'Time')]
        ts = self.df.apply(row_to_timestamp, axis=1)
        self.df['Timestamp'] = ts - ts[0]

        self.train_ratio = train_ratio
        self.n_train = int(np.ceil(len(self.df) * train_ratio))
        self.n_val = len(self.df) - self.n_train
        self.df_train = self.df.iloc[:self.n_train]
        self.df_val = self.df.iloc[self.n_train:]

        self.prices_train = self.df_train[self.prices_names].to_numpy()
        self.prices_val = self.df_val[self.prices_names].to_numpy()
        self.timestamps_train = self.df_train['Timestamp'].to_numpy()[..., None]
        self.timestamps_val = self.df_val['Timestamp'].to_numpy()[..., None]
        self.batch_size = batch_size
        self.min_inp_size = min_inp_size
        self.max_inp_size = max_inp_size
        self.min_inp_zeros = min_inp_zeros
        self.max_inp_zeros_rate = max_inp_zeros_rate

        self.rng = np.random.default_rng()

    def _gen_mask_mixed(self, inp_size: int, n_zeros: int, force_last: bool) -> np.ndarray:
        mask = np.full(inp_size, False)
        n_last = 0
        if force_last:
            n_last_min = self.min_inp_zeros
            n_last_max = max(n_zeros // 2, n_last_min)
            n_last = self.rng.integers(n_last_min, n_last_max, endpoint=True)
            mask[-n_last:] = True
            n_zeros -= n_last
        if n_zeros > 0:
            mask_inds = 1 + self.rng.choice(inp_size - n_last - 1, n_zeros, replace=False)
            mask[mask_inds] = True
        return mask

    def _get_it(self, prices: np.ndarray, timestamps: np.ndarray, n_iter: int, with_tensor: bool) -> BatchGenType:
        n = len(prices)

        for i in range(n_iter):
            inp_size = self.rng.integers(self.min_inp_size, self.max_inp_size, endpoint=True)
            n_max = n - inp_size
            assert 0 <= n_max
            inp_batch, tgt_batch, div_batch = [], [], []
            for _ in range(self.batch_size):
                ind = np.random.randint(0, n_max)
                inds = slice(ind, ind + inp_size)
                ps, ts = prices[inds], timestamps[inds]
                ps = ps / ps[0]
                max_inp_zeros = int(np.ceil(inp_size * self.max_inp_zeros_rate))
                n_zeros = self.rng.integers(self.min_inp_zeros, max_inp_zeros, endpoint=True)

                mask = self._gen_mask_mixed(inp_size, n_zeros, force_last=i % 2 == 1)
                div = discount_from_mask(mask)
                div = div.reshape((len(div), 1))

                ps_target = ps.copy()
                ps[mask] = 0
                ps_target[~mask] = 0
                tsps = np.concatenate([ts, ps], axis=1)
                inp_batch.append(tsps)
                tgt_batch.append(ps_target)
                div_batch.append(div)

            inp_batch, tgt_batch, div_batch = np.stack(inp_batch), np.stack(tgt_batch), np.stack(div_batch)
            inp_batch_t, tgt_batch_t, div_batch_t = None, None, None
            if with_tensor:
                inp_batch_t, tgt_batch_t, div_batch_t = torch.from_numpy(inp_batch), torch.from_numpy(tgt_batch), torch.from_numpy(div_batch)
                inp_batch_t, tgt_batch_t, div_batch_t = inp_batch_t.type(torch.float32), tgt_batch_t.type(torch.float32), div_batch_t.type(torch.float32)
            res = BatchResType(
                inp=inp_batch, tgt=tgt_batch, div=div_batch,
                inp_t=inp_batch_t, tgt_t=tgt_batch_t, div_t=div_batch_t)
            yield res

    def get_train_it(self, n_iter: int, with_tensor: bool = False) -> BatchGenType:
        for res in self._get_it(self.prices_train, self.timestamps_train, n_iter, with_tensor):
            yield res

    def get_val_it(self, n_iter: int, with_tensor: bool = False) -> BatchGenType:
        for res in self._get_it(self.prices_val, self.timestamps_val, n_iter, with_tensor):
            yield res

