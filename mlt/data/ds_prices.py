import os.path
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


def discount_from_mask(mask: np.ndarray, ts: np.ndarray) -> np.ndarray:
    n = len(mask)
    ts = ts.reshape(n).astype(int)
    res = np.ones(n)
    res[1:] = ts[1:] - ts[:-1]
    # For the sake of consistency, we never predict 0 time price
    res[0] = ts[1] - ts[0]
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
                    # res[ind1 - 1 - i] = i + 1
                    res[ind1 - 1 - i] = ts[ind1] - ts[ind1 - 1 - i]
            elif ind1 == n:
                assert ind > 0
                for i in range(off):
                    # res[ind + i] = i + 1
                    res[ind + i] = ts[ind + i] - ts[ind - 1]
            else:
                # j, up = 0, True
                # for i in range(off):
                #     if up:
                #         j += 1
                #         off2 = off // 2
                #         if j > off2:
                #             j = off2 + off % 2
                #             up = False
                #     else:
                #         j -= 1
                #     res[ind + i] = j
                for i in range(off):
                    dtl = ts[ind + i] - ts[ind - 1]
                    dtr = ts[ind1] - ts[ind + i]
                    res[ind + i] = min(dtl, dtr)
            ind = ind1
        else:
            ind += 1
    res = np.sqrt(res)
    return res


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).type(torch.float32)


class BatchResType:
    prices: list[np.ndarray]
    timestamp: list[np.ndarray]
    mask: Optional[list[np.ndarray]]
    discount: Optional[list[np.ndarray]]
    n_batch: int
    n_seq: int
    n_prices: int

    def __init__(self, prices: list[np.ndarray], timestamp: list[np.ndarray],
                 mask: Optional[list[np.ndarray]] = None, discount: Optional[list[np.ndarray]] = None):
        self.prices = prices
        self.timestamp = timestamp
        self.mask = mask
        self.discount = discount
        self.n_batch = len(self.prices)
        assert self.n_batch != 0 and self.n_batch == len(self.timestamp)
        if self.mask is not None:
            assert self.n_batch == len(self.mask) and self.n_batch == len(self.discount)
        prices0 = self.prices[0]
        assert prices0.ndim == 2
        self.n_seq, self.n_prices = prices0.shape

    def get_masked_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.mask is not None and self.discount is not None
        inp, tgt = [], []
        for prices, timestamp, mask in zip(self.prices, self.timestamp, self.mask):
            prices_inp, prices_tgt = prices.copy(), prices.copy()
            prices_inp[mask] = 0
            prices_tgt[~mask] = 0
            psts = np.concatenate([timestamp, prices_inp], axis=1)
            inp.append(psts)
            tgt.append(prices_tgt)
        inp, tgt, div = np.stack(inp), np.stack(tgt), np.stack(self.discount)
        div = div.reshape((self.n_batch, self.n_seq, 1))
        inp, tgt, div = to_tensor(inp), to_tensor(tgt), to_tensor(div)
        return inp, tgt, div

    def get_last_masked_tensors(self, last_zeros: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_zeros = len(last_zeros)
        timestamp, prices = self.timestamp[0].reshape((-1, 1)), self.prices[0]
        inp = np.concatenate([timestamp, prices], axis=1)
        inp = np.tile(inp, [n_zeros, 1, 1])
        tgt = np.tile(prices, [n_zeros, 1, 1])
        div = np.ones((n_zeros, self.n_seq, 1))
        mask = np.full(inp.shape[1], False)
        for iz, nz in enumerate(last_zeros):
            inp[iz, -nz:, 1:] = 0
            tgt[iz, :-nz] = 0
            m = mask.copy()
            m[-nz:] = True
            disc = discount_from_mask(m, timestamp)
            # print(disc[-max(last_zeros):])
            div[iz] = disc[..., None]
        inp, tgt, div = to_tensor(inp), to_tensor(tgt), to_tensor(div)
        return inp, tgt, div


BatchGenType = Generator[BatchResType, None, None]


class MaskGenCfg:
    mid_min_ratio: float
    mid_max_ratio: float
    last_min_ratio: float
    last_max_ratio: float
    mid_min_num: int
    mid_max_num: int
    last_min_num: int
    last_max_num: int

    def __init__(self, mid_min_ratio: float = 0, mid_max_ratio: float = 0,
                last_min_ratio: float = 0, last_max_ratio: float = 0,
                mid_min_num: int = 0, mid_max_num: int = 0,
                last_min_num: int = 0, last_max_num: int = 0
            ):
        assert mid_min_ratio == 0 or mid_min_num == 0, f'Failed: mid_min_ratio={mid_min_ratio:.3f} == 0 or mid_min_num={mid_min_num} == 0'
        assert mid_max_ratio == 0 or mid_max_num == 0, f'Failed: mid_max_ratio={mid_max_ratio:.3f} == 0 or mid_max_num={mid_max_num} == 0'
        assert last_min_ratio == 0 or last_min_num == 0, f'Failed: last_min_ratio={last_min_ratio:.3f} == 0 or last_min_num={last_min_num} == 0'
        assert last_max_ratio == 0 or last_max_num == 0, f'Failed: last_max_ratio={last_max_ratio:.3f} == 0 or last_max_num={last_max_num} == 0'
        self.mid_min_ratio = mid_min_ratio
        self.mid_max_ratio = mid_max_ratio
        self.last_min_ratio = last_min_ratio
        self.last_max_ratio = last_max_ratio
        self.mid_min_num = mid_min_num
        self.mid_max_num = mid_max_num
        self.last_min_num = last_min_num
        self.last_max_num = last_max_num

    def get_ranges(self, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
        mid_min_num = self.mid_min_num or int(np.ceil(self.mid_min_ratio * n))
        mid_max_num = self.mid_max_num or int(np.ceil(self.mid_max_ratio * n))
        last_min_num = self.last_min_num or int(np.ceil(self.last_min_ratio * n))
        last_max_num = self.last_max_num or int(np.ceil(self.last_max_ratio * n))
        assert 0 <= mid_min_num <= mid_max_num
        assert 0 <= last_min_num <= last_max_num
        assert mid_max_num + last_max_num < n - 1
        return (mid_min_num, mid_max_num), (last_min_num, last_max_num)


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
    min_timestamp_diff: float
    batch_size: int
    min_inp_size: int
    max_inp_size: int
    rng: np.random.Generator

    def __init__(self, fpath: Path, batch_size: int, min_inp_size: int, max_inp_size: int, train_ratio: float = 0.9):
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
        ts = self.df['Timestamp'].to_numpy()
        self.min_timestamp_diff = (ts[1:] - ts[:-1]).min()

        self.prices_train = self.df_train[self.prices_names].to_numpy()
        self.prices_val = self.df_val[self.prices_names].to_numpy()
        self.timestamps_train = self.df_train['Timestamp'].to_numpy()[..., None]
        self.timestamps_val = self.df_val['Timestamp'].to_numpy()[..., None]
        self.batch_size = batch_size
        self.min_inp_size = min_inp_size
        self.max_inp_size = max_inp_size

        self.rng = np.random.default_rng()

    def _gen_mask(self, inp_size: int, mid_range: tuple[int, int], last_range: tuple[int, int]):
        mask = np.full(inp_size, False)
        n_mid, n_last = self.rng.integers(*mid_range, endpoint=True), self.rng.integers(*last_range, endpoint=True)
        if n_last > 0:
            mask[-n_last:] = True
        if n_mid > 0:
            mask_inds = 1 + self.rng.choice(inp_size - 1 - n_last, n_mid, replace=False)
            mask[mask_inds] = True
        return mask

    def _get_it(self, prices: np.ndarray, timestamps: np.ndarray, n_iter: int, cfgm: Optional[MaskGenCfg]) -> BatchGenType:
        n = len(prices)
        gen_mask = cfgm is not None
        cfgm = cfgm or MaskGenCfg()

        for i in range(n_iter):
            inp_size = self.rng.integers(self.min_inp_size, self.max_inp_size, endpoint=True)
            mid_range, last_range = cfgm.get_ranges(inp_size)
            n_max = n - inp_size
            assert 0 <= n_max
            prices_batch, timestamp_batch = [], []
            mask_batch, discount_batch = ([], []) if gen_mask else (None, None)
            for _ in range(self.batch_size):
                ind = np.random.randint(0, n_max)
                inds = slice(ind, ind + inp_size)
                ps, ts = prices[inds], timestamps[inds]
                ps, ts = ps / ps[0], (ts - ts[0]) / self.min_timestamp_diff
                prices_batch.append(ps)
                timestamp_batch.append(ts)

                if gen_mask:
                    mask = self._gen_mask(inp_size, mid_range, last_range)
                    discount = discount_from_mask(mask, ts).reshape((inp_size, 1))
                    mask_batch.append(mask)
                    discount_batch.append(discount)

            res = BatchResType(prices=prices_batch, timestamp=timestamp_batch,
                               mask=mask_batch, discount=discount_batch)
            yield res

    def get_train_it(self, n_iter: int, cfgm: Optional[MaskGenCfg] = None) -> BatchGenType:
        for res in self._get_it(self.prices_train, self.timestamps_train, n_iter, cfgm):
            yield res

    def get_val_it(self, n_iter: int, cfgm: Optional[MaskGenCfg] = None) -> BatchGenType:
        for res in self._get_it(self.prices_val, self.timestamps_val, n_iter, cfgm):
            yield res


def test_ds():
    fname = 'stocks_1.1.18-9.2.24_close.csv'
    fpath = Path(os.path.expandvars('$HOME')) / 'data/mltrade' / fname
    batch_size = 5
    min_inp_size, max_inp_size = 1 * 20, 2 * 20
    cfg_mask = MaskGenCfg(
        mid_min_num=2, mid_max_num=8,
        last_min_num=5, last_max_num=10,
    )
    ds = DsPrices(fpath, batch_size, min_inp_size, max_inp_size)
    batch_it = ds.get_train_it(5, cfg_mask)
    for i, batch in enumerate(batch_it):
        batch: BatchResType = batch
        print(f'Batch {i}')
        print('Prices:', batch.prices[0])
        print('Mask:', batch.mask[0].flatten())
        print('Discount:', batch.discount[0].flatten())


if __name__ == '__main__':
    test_ds()

